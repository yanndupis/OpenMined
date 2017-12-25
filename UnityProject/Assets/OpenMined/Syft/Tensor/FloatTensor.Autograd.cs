﻿using System;
using System.Collections.Generic;
using OpenMined.Syft.NN;
using UnityEngine;

namespace OpenMined.Syft.Tensor
{
    public partial class FloatTensor
    {

	    private bool autograd;
	    public FloatTensor Grad { get; private set; }
	    private bool keepgrads;
	    
	    // checks to see if a variable has accumulated all the gradients it should before it backprops
	    public bool AllAutogradChildrenAccountedFor()
	    {
		    for (int i=0; i< children_counts.Count; i++)
		    {
			    if (children_counts[i] == 0 && factory.Get(children_indices[i]).autograd)
			    {
				    return false;
			    }
		    }
		    return true;
	    }
	    
	    public void Backward(FloatTensor grad = null, FloatTensor grad_origin = null)
	    {
		    //Debug.Log("Backward:" + this.id + " creation_op:" + creation_op);
		  
		    if (autograd)
		    {
			    if (grad == null)
			    {
				    Debug.Log("Grad not Found... Creating Gradient of 1s");
				    grad = this.createOnesTensorLike();
				    grad.Autograd = false;
			    }

			    if (grad_origin != null)
			    {
				    int child_index = children_indices.IndexOf(grad_origin.Id);
				    if (children_counts[child_index] > 0)
				    {
					    throw new InvalidOperationException("Can't backprop more than once.");
				    }
				    else
				    {
					    children_counts[child_index] += 1;
				    }
			    }

			    if (this.Grad == null)
			    {
				    this.Grad = grad;
				    //Debug.Log("Setting Grad Tensor Id:" + this.id);
			    }
			    else
			    {
				    if (this.Grad.id == grad.id)
				    {
					    // do nothing
					    //Debug.Log("Not Updating For Tensor Id:" + this.id);
				    }
				    else
				    {
					    //Debug.Log("Updating For Tensor Id:" + this.id);
					    //this.Grad.Zero_();
					    this.Grad.Add(grad, inline: true);
				    }

			    }

			    // grads must not have grads of their own
			    if (this.Grad.autograd == true)
			    {
				    throw new InvalidOperationException("Sorry, grads cannot have grads");
			    }
			    
				// RULES FOR AUTOGRAD:
			    // 1) if you need to use "this" for calculating a gradient, copy it first and set autograd to false (see sigmoid)
			    // 2) if you use a method in your backprop logic that doesn't hook into the dynamic graph yet, backprop
			    // will not work!!! Make sure there's a "hookautograd" function in every method you use for backprop.
			    // 3) whenever backpropping into a method where the forward prop involved a scalar (such as scalar
			    // multiplication), current implementations assume you will NOT backprop into the scalar itself.
			    // 4) Because of rule (2), do NOT use "emptyTensorCopy" at all in backprop unless you know what you're
			    // doing. 
			    // 5) I will be especially strict about Unit tests for all backprop logic as this is the most complex
			    // piece of functionality we have. Furthermore, most errors go completely undetected (not discovered
			    // by runtime errors). Autograd bugs just make convergence go slowly and sub-optimally.
			    // 6) If you use a forward propagation tensor to backprop, you MUST remember to turn off autograd
			    // when backpropagating (see "mm" below for example). Otherwise, it will cause autograd to break because
			    // whatever child you select will think it needs to wait for another gradient before backpropagating.
			    
			    // only continue backpropping if there's something to backprop into
			    // only continue backpropping if all gradients (from children) are accounted for
			    // override waiting for children if "backprop" was called on this variable directly
			    if (this.creators != null && this.creators.Count > 0 && (grad_origin == null || AllAutogradChildrenAccountedFor()))
			    {
				    
				    if (creation_op == "add_elem")
				    {

					    factory.Get(creators[0]).Backward(grad, this);
					    factory.Get(creators[1]).Backward(grad, this);

				    }
				    else if (creation_op == "add_scalar")
				    {
					    factory.Get(creators[0]).Backward(grad, this);
				    }
				    else if (creation_op == "copy")
				    {
					    factory.Get(creators[0]).Backward(grad, this);
				    }
				    else if (creation_op == "div_elem")
				    {
					    FloatTensor x = factory.Get(creators[0]);
					    FloatTensor y = factory.Get(creators[1]);

					    x.Backward(grad.Div(y));

					    FloatTensor y2 = y.Pow(2);
					    FloatTensor xn = x.Neg();
					    FloatTensor xny2 = xn.Div(y2);
					    FloatTensor gradxny2 = grad.Mul(xny2);
					    y.Backward(gradxny2);
				    }
				    else if (creation_op == "div_scalar")
				    {
					    factory.Get(creators[0]).Backward(grad.Div(factory.Get(creators[1]).data[0]), this);
				    }
				    else if (creation_op == "mul_elem")
				    {
					    factory.Get(creators[0]).Backward(grad.Mul(factory.Get(creators[1])), this);
					    factory.Get(creators[1]).Backward(grad.Mul(factory.Get(creators[0])), this);
				    }
				    else if (creation_op == "mul_scalar")
				    {
					    factory.Get(creators[0]).Backward(grad.Mul(factory.Get(creators[1]).data[0]), this);
				    }
				    else if (creation_op == "mm")
				    {
					    FloatTensor x = factory.Get(creators[1]).Transpose();
					    x.autograd = false;

					    FloatTensor y = factory.Get(creators[0]).Transpose();
					    y.autograd = false;
					    
					    factory.Get(creators[0]).Backward(grad.MM(x), this);
					    factory.Get(creators[1]).Backward(y.MM(grad), this);
				    }
				    else if (creation_op == "neg")
				    {
					    factory.Get(creators[0]).Backward(grad.Neg(), this);
				    }
				    else if (creation_op == "pow_scalar")
				    {

					    FloatTensor x = factory.Get(creators[0]).Copy();
					    x.autograd = false;
					    
					    factory.Get(creators[0]).Backward(x.Mul(grad).Mul(factory.Get(creators[1]).Data[0]), this);
				    }
				    else if (creation_op == "relu")
                    {

                        FloatTensor c = this.Copy();
                        c.autograd = false;
                        factory.Get(creators[0]).Backward(grad, this);

                    }
				    else if (creation_op == "sub_elem")
				    {
					    factory.Get(creators[0]).Backward(grad, this);
					    factory.Get(creators[1]).Backward(grad.Neg(), this);
				    }
				    else if (creation_op == "sub_scalar")
				    {
					    factory.Get(creators[0]).Backward(grad, this);
				    }
				    else if (creation_op == "sigmoid")
				    {
					    FloatTensor self_nograd = this.Copy();
					    self_nograd.autograd = false;
					    
					    factory.Get(creators[0]).Backward(self_nograd.Neg().Add((float) 1).Mul(self_nograd).Mul(grad), this);
				    }
				    else if (creation_op == "transpose")
				    {
					    factory.Get(creators[0]).Backward(grad.Transpose());
				    }
				    else if (creation_op == "tanh")
				    {
					    FloatTensor c = this.Copy();
					    c.autograd = false;

					    factory.Get(creators[0]).Backward(c.Pow(2).Neg().Add(1f).Mul(grad), this);
				    }
				    else if (creation_op.Contains("softmax-"))
				    {

					    FloatTensor c = this.Copy();
					    c.autograd = false;
					    var dim = int.Parse(creation_op.Split('-')[1]);
					    factory.Get(creators[0]).Backward(Functional.SoftmaxGradient(this, grad, dim), this);

				    }
				    else
				    {
					    Debug.Log("Autograd couldn't find matching operation for:" + creation_op);   
				    }
			    }
		    }
		    else
		    {
			    Debug.Log("Autograd off - skipping backprop...");
		    }
	    }

    }
}