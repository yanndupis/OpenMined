namespace OpenMined.Syft.Tensor
{
    public partial class BaseTensorGeneric<T>
    {
        private bool isEncrypted;

        public bool IsEncrypted
        {
            get { return isEncrypted; }
        }
    }
}