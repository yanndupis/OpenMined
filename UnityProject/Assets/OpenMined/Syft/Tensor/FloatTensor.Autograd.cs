using System;
using System.Collections.Generic;
using UnityEngine;

namespace OpenMined.Syft.Tensor
{
    public partial class FloatTensor
    {
	    private bool autograd;
	    public FloatTensor Grad { get; private set; }
	    private bool keepgrads;
	    
	    public void Backward(FloatTensor grad = null, FloatTensor grad_origin = null)
	    {

		    if (autograd)
		    {
			    if (grad == null)
			    {
				    Debug.Log("Grad not Found... Creating Gradient of 1s");
				    grad = this.controller.createOnesTensorLike(this);
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
			    }
			    else
			    {
				    if (this.Grad.id == grad.id)
				    {
						// do nothing
				    }
				    else
				    {
					    this.Grad.Zero_();
					    this.Grad.Add(grad, true);    
				    }
				    
			    }

			    // grads must not have grads of their own
			    if (this.Grad.autograd == true)
			    {
				    throw new InvalidOperationException("Sorry, grads cannot have grads");
			    }

			    // only continue backpropping if there's something to backprop into
			    // only continue backpropping if all gradients (from children) are accounted for
			    // override waiting for children if "backprop" was called on this variable directly
			    if (this.creators != null && this.creators.Count > 0 && (grad_origin == null || AllChildrenGradsAccountedFor()))
			    {
				    if (creation_op == "add_elem")
				    {
					
					    controller.getTensor(creators[0]).Backward(grad, this);
					    controller.getTensor(creators[1]).Backward(grad, this);

				    }
				    else if (creation_op == "add_scalar")
				    {
					    controller.getTensor(creators[0]).Backward(grad, this);
				    }
				    else if (creation_op == "copy")
				    {
					    controller.getTensor(creators[0]).Backward(grad, this);
				    }
				    else if (creation_op == "div_elem")
				    {
					    FloatTensor x = controller.getTensor(creators[0]);
					    FloatTensor y = controller.getTensor(creators[1]);
					    
						x.Backward(grad.Div(y));

					    FloatTensor y2 = y.Pow(2);
					    FloatTensor xn = x.Neg();
					    FloatTensor xny2 = xn.Div(y2);
					    FloatTensor gradxny2 = grad.Mul(xny2);
					    y.Backward(gradxny2);	
				    }
				    else if (creation_op == "div_scalar")
				    {
					    controller.getTensor(creators[0]).Backward(grad.Div(controller.getTensor(creators[1]).data[0]), this);
				    }
				    else if (creation_op == "mul_elem")
				    {
					    controller.getTensor(creators[0]).Backward(grad.Mul(controller.getTensor(creators[1])), this);
					    controller.getTensor(creators[1]).Backward(grad.Mul(controller.getTensor(creators[0])), this);
				    }
				    else if (creation_op == "mul_scalar")
				    {
					    controller.getTensor(creators[0]).Backward(grad.Mul(controller.getTensor(creators[1]).data[0]), this);
				    }
				    else if (creation_op == "mm")
				    {
					    controller.getTensor(creators[0]).Backward(grad.MM(controller.getTensor(creators[1]).Transpose()), this);
					    controller.getTensor(creators[1]).Backward(controller.getTensor(creators[0]).Transpose().MM(grad), this);
				    }
				    else if (creation_op == "pow_scalar")
				    {
					    FloatTensor self_nograd = controller.getTensor(creators[0]);
					    controller.getTensor(creators[0]).Backward(self_nograd.Mul(grad).Mul(controller.getTensor(creators[1]).Data[0]), this);
				    }
				    else if (creation_op == "sub_elem")
				    {
					    controller.getTensor(creators[0]).Backward(grad, this);
					    controller.getTensor(creators[1]).Backward(grad.Neg(), this);
				    }
				    else if (creation_op == "sub_scalar")
				    {
					    controller.getTensor(creators[0]).Backward(grad, this);
				    }
				    else if (creation_op == "sigmoid")
				    {
					    controller.getTensor(creators[0]).Backward(this.Neg().Add((float) 1).Mul(this).Mul(grad), this);
				    }
				    else if (creation_op == "transpose")
				    {
					    controller.getTensor(creators[0]).Backward(grad.Transpose());
				    }
			    }
		    }
	    }
    }
}