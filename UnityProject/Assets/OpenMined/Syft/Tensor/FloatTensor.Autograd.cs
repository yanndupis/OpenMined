using System;
using System.Collections.Generic;

namespace OpenMined.Syft.Tensor
{
	public partial class FloatTensor
	{

		private bool autograd;
		private bool keepgrads;

		private List<FloatTensor> creators;
		private string creation_op;
		private Dictionary<int, int> children;

		private FloatTensor grad;

	

		public void InitAutograd() {
//			if(!autograd) {
				autograd=true;
				creators = new List<FloatTensor> ();
				children = new Dictionary<int, int> ();
//			}
		}

		public bool AllChildrenGradsAccountedFor() {
			foreach(var item in children)
			{
				if (item.Value == 0) {
					return false;
				}
			}
			return true;
		}

		public void HookAutograd(ref FloatTensor result, ref FloatTensor x, string creation_op) {
		
			if (autograd) {

				result.InitAutograd ();
				result.creators.Add (this);
				result.creators.Add (x);
				result.creation_op = creation_op;

				children.Add (result.Id, 0);
				x.children.Add (result.Id, 0);

			}
		
		}

		public void Backward(FloatTensor grad = null, FloatTensor grad_origin=null) {

			if (autograd) {
				if (grad == null) {
					grad = this.ctrl.createOnesTensorLike (this);
					grad.Autograd = false;
				}

				if (grad_origin != null) {
					if (children[grad_origin.Id] > 0) {
						throw new InvalidOperationException ("Can't backprop more than once.");
					} else {
						children [grad_origin.Id] += 1;
					}
				}	

				if (this.grad == null) {
					this.grad = grad;
				} else {
					this.grad.Add (grad, true);
				}

				// grads must not have grads of their own
				if (this.grad.autograd == true) {
					throw new InvalidOperationException ("Sorry, grads cannot have grads");
				}

				// only continue backpropping if there's something to backprop into
				// only continue backpropping if all gradients (from children) are accounted for
				// override waiting for children if "backprop" was called on this variable directly
				if(this.creators != null && this.creators.Count > 0 && (grad_origin == null || AllChildrenGradsAccountedFor())) {
					if (creation_op == "add_elem") {

						creators [0].Backward (grad, this);
						creators [1].Backward (grad, this);

					} else if (creation_op == "mul_elem") {
						creators [0].Backward (grad.Mul (creators [1]), this);
						creators [1].Backward (grad.Mul (creators [0]), this);
					} else if (creation_op == "div_elem") {
						creators [0].Backward (grad.Div (creators [1]), this);
						creators [1].Backward (grad.Div (creators [0]), this);
					} else if (creation_op == "sub_elem") {
						creators [0].Backward (grad, this);
						creators [1].Backward (grad.Neg(), this);
					}

//					if (!keepgrads) {
//						ctrl.RemoveTensor (grad.id);
//					}
				}
			}



		}

	}
}
