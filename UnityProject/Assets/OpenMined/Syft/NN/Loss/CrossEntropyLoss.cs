using OpenMined.Syft.Tensor;
using OpenMined.Network.Controllers;

namespace OpenMined.Syft.Layer.Loss
{
    public class CrossEntropyLoss: Loss
	{
		public CrossEntropyLoss (SyftController controller)
		{
			init("crossentropyloss");

			#pragma warning disable 420
			id = System.Threading.Interlocked.Increment(ref nCreated);
			controller.addModel(this);

		}
        public override FloatTensor Forward(FloatTensor prediction, FloatTensor target)
		{
			// Note: prediction should be logits, basically pre-softmax. This method applies softmax first. 
			// TODO check shapes 

			FloatTensor softmax = prediction.Softmax();
            FloatTensor output = ((target.Mul(softmax.Log1p())).Sum()).Mul(-1);
			return output;
		}

	}
}

