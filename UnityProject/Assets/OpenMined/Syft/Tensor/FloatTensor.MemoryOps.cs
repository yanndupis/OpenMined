using UnityEngine;

namespace OpenMined.Syft.Tensor
{
    // TODO: Implement move data to GPU to CPU etc. in this file
    public partial class FloatTensor
    {   
        private bool dataOnGpu;

        private ComputeBuffer dataBuffer;
        private ComputeBuffer shapeBuffer;

        public bool DataOnGpu => dataOnGpu;

        public ComputeBuffer DataBuffer
        {
            get { return dataBuffer; }
            set { dataBuffer = value; }
        }
        
		public bool Gpu(ComputeShader _shader)
        {
            if (dataOnGpu || !SystemInfo.supportsComputeShaders) return false;
			shader = _shader;

            CopyCputoGpu();
            EraseCpu();
            return true;
        }

        public void Cpu()
        {
            if (!dataOnGpu) return;
            CopyGpuToCpu();
            EraseGpu();
        }
        
        private void CopyGpuToCpu()
        {
            data = new float[size];
            dataBuffer.GetData(Data);
        }
        
        private void CopyCputoGpu()
        {
			initShaderKernels ();

            dataBuffer = new ComputeBuffer(size, sizeof(float));
            shapeBuffer = new ComputeBuffer(shape.Length, sizeof(int));

            dataBuffer.SetData(Data);	
            shapeBuffer.SetData(shape);
            
            dataOnGpu = true;
        }

        private void EraseCpu()
        {
            data = null;
        }
        
        private void EraseGpu()
        {
            dataBuffer.Release();
            shapeBuffer.Release();
            dataOnGpu = false;
        }
    }
}
