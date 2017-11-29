using UnityEngine;

namespace OpenMined.Syft.Shaders
{
    public class FloatTensorShader
    {
        private static ComputeShader shader = null;
        
        [SerializeField]
        private static int absKernel;
        [SerializeField]
        private static int absKernel_;
        [SerializeField]
        private static int addScalarKernel_;
        [SerializeField]
        private static int addElemKernel_;
        [SerializeField]
        private static int addScalarKernel;
        [SerializeField]
        private static int addElemKernel;
        [SerializeField]
        private static int addMMKernel_;
        [SerializeField]
        private static int ceilKernel;
        [SerializeField]
        private static int floorKernel_;
        [SerializeField]
        private static int mulScalarKernel_;
        [SerializeField]
        private static int mulElemKernel_;
        [SerializeField]
        private static int mulScalarKernel;
        [SerializeField]
        private static int mulElemKernel;
        [SerializeField]
        private static int negateKernel;
        [SerializeField]
        private static int powKernel;
        [SerializeField]
        private static int powKernel_;
        [SerializeField]
        private static int sigmoidKernel_;
        [SerializeField]
        private static int subElemKernel;
        [SerializeField]
        private static int tanhKernel;
        [SerializeField]
        private static int zeroKernel_;

        public static void InitWithShader(ComputeShader _shader)
        {
            shader = _shader;
            
            absKernel = shader.FindKernel("Abs");
            absKernel_ = shader.FindKernel("Abs_");
            addScalarKernel_ = shader.FindKernel("AddScalar_");
            addElemKernel_ = shader.FindKernel("AddElem_");
            addScalarKernel = shader.FindKernel("AddScalar");
            addElemKernel = shader.FindKernel("AddElem");
            addMMKernel_ = shader.FindKernel("AddMM_");
            ceilKernel = shader.FindKernel("Ceil");
            floorKernel_ = shader.FindKernel("Floor_");
            mulScalarKernel_ = shader.FindKernel("MulScalar_");
            mulElemKernel_ = shader.FindKernel("MulElem_");
            mulScalarKernel = shader.FindKernel("MulScalar");
            mulElemKernel = shader.FindKernel("MulElem");
            negateKernel = shader.FindKernel("Negate");
            powKernel = shader.FindKernel("Pow");
            powKernel_ = shader.FindKernel("Pow_");
            sigmoidKernel_ = shader.FindKernel("Sigmoid_");
            subElemKernel = shader.FindKernel("SubElem");
            tanhKernel = shader.FindKernel("Tanh");
            zeroKernel_ = shader.FindKernel("Zero_");
        }
        
        public static ComputeShader Shader => shader;

        public static int AbsKernel => absKernel;

        public static int AbsKernel_ => absKernel_;

        public static int AddScalarKernel => addScalarKernel_;

        public static int AddElemKernel => addElemKernel_;

        public static int AddScalarKernel_ => addScalarKernel;

        public static int AddElemKernel_ => addElemKernel;

        public static int AddMMKernel_ => addMMKernel_;

        public static int CeilKernel => ceilKernel;

        public static int FloorKernel_ => floorKernel_;

        public static int MulScalarKernel_ => mulScalarKernel_;

        public static int MulElemKernel_ => mulElemKernel_;

        public static int MulScalarKernel => mulScalarKernel;

        public static int MulElemKernel => mulElemKernel;

        public static int NegateKernel => negateKernel;

        public static int PowKernel => powKernel;

        public static int PowKernel_ => powKernel_;

        public static int SigmoidKernel_ => sigmoidKernel_;

        public static int SubElemKernel => subElemKernel;

        public static int TanhKernel => tanhKernel;

        public static int ZeroKernel_ => zeroKernel_;
    }
}