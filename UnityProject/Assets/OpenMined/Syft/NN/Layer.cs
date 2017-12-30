using System;
using OpenMined.Syft.Tensor;
using OpenMined.Network.Utils;
using OpenMined.Network.Controllers;

namespace OpenMined.Syft.Layer
{
    public abstract class Layer: Model
    {

        public abstract FloatTensor Forward (FloatTensor input);

        protected override string ProcessForwardMessage(Command msgObj, SyftController ctrl)
        {
            var input = ctrl.floatTensorFactory.Get(int.Parse(msgObj.tensorIndexParams[0]));
            if (input.Autograd)
            {
                var result = this.Forward(input);
                return result.Id + "";
            }
            else
            {
                throw new Exception("Input to Model object must have autograd == true but autograd == false!!!");
            }
        }
    }
}

