using System;
using System.Diagnostics;
using JetBrains.Annotations;
using OpenMined.Syft.Tensor;
using OpenMined.Network.Utils;
using OpenMined.Network.Controllers;
using OpenMined.Syft.Optim;

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
        
        public string Fit(FloatTensor input, FloatTensor target, Loss.Loss criterion, SGD optimizer, int iter)
        {
            controller.Log("Fitting...");
            
            FloatTensor grad = null;
            FloatTensor loss = null;
            FloatTensor pred = null;
            
            for (int i = 0; i < iter; i++)
            {
                pred = Forward(input);
                loss = criterion.Forward(pred, target);
                
                if (grad == null)
                {
                    grad = loss.createOnesTensorLike();
                    grad.Autograd = false;
                }
                
                loss.Backward(grad);
                
                optimizer.Step();
                
            }
            
            if(loss != null)
                return loss.Id.ToString();
            
            return "Loss is Null";
        }
        
        protected override string ProcessMessageAsLayerOrLoss (Command msgObj, SyftController ctrl)
        {
            
            switch (msgObj.functionCall)
            {
                case "fit":
                {
                    FloatTensor input = ctrl.floatTensorFactory.Get(int.Parse(msgObj.tensorIndexParams[0]));
                    FloatTensor target = ctrl.floatTensorFactory.Get(int.Parse(msgObj.tensorIndexParams[1]));
                    Loss.Loss criterion = ctrl.getLoss(int.Parse(msgObj.tensorIndexParams[2]));
                    SGD optim = ctrl.getOptimizer(int.Parse(msgObj.tensorIndexParams[3]));
                    int iters = int.Parse(msgObj.tensorIndexParams[4]);

                    return Fit(input, target, criterion, optim, iters);
                }
            }

            return ProcessMessageAsLayerObject(msgObj, ctrl);
        }
		
        protected virtual string ProcessMessageAsLayerObject (Command msgObj, SyftController ctrl) 
        {   
            return "Model.processMessage not Implemented:" + msgObj.functionCall;
        }
    }
}

