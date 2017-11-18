using System;

namespace OpenMined.Syft.Tensor
{
    public partial class FloatTensor : IDisposable
    {

        private bool disposed = false;

        public void Dispose()
        { 
            Dispose(true);
            GC.SuppressFinalize(this);           
        }

        protected virtual void Dispose(bool disposing)
        {
            if (disposed)
                return; 

            if (disposing)
            {
                data = null;
                shape = null;
                strides = null;
            }

            disposed = true;
        }

        ~FloatTensor()
        {
            Dispose(false);
        }
    }
}