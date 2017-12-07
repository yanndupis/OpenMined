using System;
using System.Runtime.Remoting.Messaging;
using NUnit.Framework.Constraints;
using UnityEngine;
using OpenMined.Network.Utils;
using OpenMined.Network.Controllers;
using System.Collections.Generic;

namespace OpenMined.Syft.Tensor
{
    public partial class FloatTensor
    {
// Should we put a check incase this variable overflows?
        private static volatile int nCreated = 0;

        public SyftController ctrl;

        private float[] data;
        private long[] strides;
        private int[] shape;
        private int size;

        private int id;

//Making this public for now. Usage of below functions might not be efficient but can help development
//private
        public long GetIndex(params long[] indices)
        {
            long offset = 0;
            for (int i = 0; i < indices.Length; ++i)
            {
                if (indices[i] >= shape[i] || indices[i] < 0)
                    throw new IndexOutOfRangeException();
                offset += indices[i] * strides[i];
            }
            return offset;
        }

//private
        public long GetIndex(params int[] indices)
        {
            var long_indices = Array.ConvertAll(indices, item => (long) item);
            return GetIndex(long_indices);
        }

//private
        public long[] GetIndices(long index)
        {
            var idx = index;
            long[] indices = new long[Shape.Length];
            for (int i = 0; i < Shape.Length; ++i)
            {
                indices[i] = (idx - (idx % (strides[i]))) / strides[i];
                idx -= indices[i] * strides[i];
            }
            return indices;
        }

        public float[] Data
        {
            get { return data; }
        }

        public int[] Shape
        {
            get { return shape; }
        }

        public int Size
        {
            get { return size; }
        }

        public int Id
        {
            get { return id; }

            set { id = value; }
        }

        public bool Autograd
        {
            get { return autograd; }

            set { autograd = value; }
        }

        public static int CreatedObjectCount
        {
            get { return nCreated; }
        }

        public long[] Strides
        {
            get { return strides; }
        }

        // parameters are overrides
        public FloatTensor Copy()
        {
            FloatTensor copy = new FloatTensor(ctrl,
                _shape: this.shape,
                _data: data,
                _dataBuffer: dataBuffer,
                _shapeBuffer: shapeBuffer,
                _shader: shader,
                _copyData: true,
                _dataOnGpu: dataOnGpu,
                _autograd: autograd,
                _keepgrads: keepgrads,
                _creation_op: creation_op);
            return copy;
        }

        public FloatTensor(SyftController _ctrl,
            int[] _shape,
            float[] _data = null,
            ComputeBuffer _dataBuffer = null,
            ComputeBuffer _shapeBuffer = null,
            ComputeShader _shader = null,
            bool _copyData = true,
            bool _dataOnGpu = false,
            bool _autograd = false,
            bool _keepgrads = false,
            string _creation_op = null)
        {
            ctrl = _ctrl;

            dataOnGpu = _dataOnGpu;
            autograd = _autograd;
            keepgrads = _keepgrads;
            creation_op = _creation_op;

            if (autograd)
            {
                InitAutograd();
            }

            // First: check that shape is valid.
            if (_shape == null || _shape.Length == 0)
            {
                throw new InvalidOperationException("Tensor shape can't be an empty array.");
            }

            // Second: since shape is valid, let's save it
            shape = (int[]) _shape.Clone();

            setStridesAndCheckShape();

            // Third: let's see what kind of data we've got. We should either have
            // a GPU ComputeBuffer or a data[] object. 
            if (_data != null && _shapeBuffer == null && _dataBuffer == null)
            {
                InitCpu(_data: _data, _copyData: _copyData);
            }
            else if (_dataBuffer != null && _shapeBuffer != null && SystemInfo.supportsComputeShaders && _data == null)
            {
                // looks like we have GPU data being passed in... initialize a GPU tensor.

                InitGpu(_shader, _dataBuffer, _shapeBuffer, _copyData);
            }
            else
            {
                // no data seems to be passed in... or its got missing stuff

                // if CPU works... go with that
                if (_data != null)
                {
                    InitCpu(_data, _copyData);
                }
                else if (_dataBuffer != null && _shader != null)
                {
                    if (SystemInfo.supportsComputeShaders)
                    {
                        // seems i'm just missing a shape buffer - no biggie
                        shapeBuffer = new ComputeBuffer(shape.Length, sizeof(int));
                        shapeBuffer.SetData(shape);

                        InitGpu(_shader, _dataBuffer, _shapeBuffer, _copyData);
                    }
                    else
                    {
                        throw new InvalidOperationException(
                            "You seem to be trying to create a GPU tensor without having access to a GPU...");
                    }
                }
                else
                {
                    // nothing else seems to work - i suppose i'm just supposed to initialize an empty tensor.
                    long acc = 1;
                    for (var i = shape.Length - 1; i >= 0; --i)
                    {
                        acc *= shape[i];
                    }

                    if (_dataOnGpu)
                    {
                        _shapeBuffer = new ComputeBuffer(shape.Length, sizeof(int));
                        _shapeBuffer.SetData(shape);

                        _dataBuffer = new ComputeBuffer(size, sizeof(float));

                        InitGpu(_shader: _shader, _dataBuffer: _dataBuffer, _shapeBuffer: _shapeBuffer,
                            _copyData: false);
                        this.Zero_();
                    }
                    else
                    {
                        _data = new float[acc];

                        InitCpu(_data, false);
                    }
                }
            }

//			// Lastly: let's set the ID of the tensor.
//			// IDEs might show a warning, but ref and volatile seems to be working with Interlocked API.
            id = System.Threading.Interlocked.Increment (ref nCreated);
            
            ctrl.addTensor(this);
            if (SystemInfo.supportsComputeShaders && shader == null)
            {
                shader = ctrl.GetShader();
            }
//
//
        }

        public void InitCpu(float[] _data, bool _copyData)
        {
            // looks like we have CPU data being passed in... initialize a CPU tensor.
            dataOnGpu = false;

            if (_copyData)
            {
                data = (float[]) _data.Clone();
            }
            else
            {
                data = _data;
            }
        }

        public void InitGpu(ComputeShader _shader, ComputeBuffer _dataBuffer, ComputeBuffer _shapeBuffer,
            bool _copyData)
        {
            if (!SystemInfo.supportsComputeShaders)
                throw new NotSupportedException("Shaders are not supported on the host machine");
            
            // First: we need to check that we have a shader
            if (_shader != null)
            {
                shader = _shader;
                initShaderKernels();
            }
            else
            {
                throw new FormatException("You tried to initialize a GPU tensor without access to a shader or gpu.");
            }


            if (_copyData)
            {
                float[] temp_data = new float[_dataBuffer.count];
                int[] temp_shape = new int[shape.Length];

                _dataBuffer.GetData(temp_data);
                _shapeBuffer.GetData(temp_shape);

                dataBuffer = new ComputeBuffer(_dataBuffer.count, sizeof(float));
                shapeBuffer = new ComputeBuffer(_shapeBuffer.count, sizeof(int));

                dataBuffer.SetData(temp_data);
                shapeBuffer.SetData(temp_shape);
            }

            // Third: let's set the tensor's size to be equal to that of the buffer
            size = _dataBuffer.count;
        }

        public void setStridesAndCheckShape()
        {
            // Third: let's initialize our strides.
            strides = new long[shape.Length];

            // Fifth: we should check that the buffer's size matches our shape.
            int acc = 1;
            for (var i = shape.Length - 1; i >= 0; --i)
            {
                strides[i] = acc;
                acc *= shape[i];
            }

            // Sixth: let's check to see that our shape and data sizes match.
            size = acc;
        }


        public float this[params long[] indices]
        {
            get { return this[GetIndex(indices)]; }
            set { this[GetIndex(indices)] = value; }
        }

        public float this[long index]
        {
            get { return Data[index]; }
            set { Data[index] = value; }
        }

        public string ProcessMessage(Command msgObj, SyftController ctrl)
        {
            switch (msgObj.functionCall)
            {
                case "abs":
                {
                    // calls the function on our tensor object
                    var result = this.Abs();
                    // returns the function call name with the OK status
                    return result.id + "";
                }
                case "abs_":
                {
                    // calls the function on our tensor object
                    this.Abs(inline: true);
                    // returns the function call name with the OK status
                    return id.ToString();
                }
                case "add_elem":
                {
                    Debug.LogFormat("add_elem");
                    var tensor_1 = ctrl.getTensor(int.Parse(msgObj.tensorIndexParams[0]));
                    var result = this.Add(tensor_1);
                    return result.id + "";
                }
                case "acos":
                {
                    var result = Acos();
                    return result.Id.ToString();
                }
                case "acos_":
                {
                    Acos(inline: true);
                    return Id.ToString();
                }
                case "asin":
                {
                    var result = Asin();
                    return result.Id.ToString();
                }
                case "asin_":
                {
                    Asin(inline: true);
                    return Id.ToString();
                }
                case "atan":
                {
                    var result = Atan();
                    return result.Id.ToString();
                }
                case "atan_":
                {
                    Atan(inline: true);
                    return Id.ToString();
                }
                case "add_elem_":
                {
                    Debug.LogFormat("add_elem_");
                    var tensor_1 = ctrl.getTensor(int.Parse(msgObj.tensorIndexParams[0]));
                    this.Add(tensor_1, inline: true);
                    return this.id + "";
                }
                case "add_scalar":
                {
                    Debug.LogFormat("add_scalar");
                    FloatTensor result = Add(float.Parse(msgObj.tensorIndexParams[0]));
                    return result.Id + "";
                }
                case "add_scalar_":
                {
                    Debug.LogFormat("add_scalar_");
                    this.Add(float.Parse(msgObj.tensorIndexParams[0]), inline: true);
                    return this.id + "";
                }
                case "addmm_":
                {
                    var tensor_1 = ctrl.getTensor(int.Parse(msgObj.tensorIndexParams[0]));
                    var tensor_2 = ctrl.getTensor(int.Parse(msgObj.tensorIndexParams[1]));
                    AddMatrixMultiply(tensor_1, tensor_2);
                    return msgObj.functionCall + ": OK";
                }
                case "addmv_":
                {
                    var tensor_1 = ctrl.getTensor(int.Parse(msgObj.tensorIndexParams[0]));
                    var tensor_2 = ctrl.getTensor(int.Parse(msgObj.tensorIndexParams[1]));
                    AddMatrixVectorProduct(tensor_1, tensor_2);
                    return msgObj.functionCall + ": OK";
                }
                case "backward":
                {
                    if (msgObj.tensorIndexParams.Length > 0)
                    {
                        var grad = ctrl.getTensor(int.Parse(msgObj.tensorIndexParams[0]));
                        Backward(grad);
                    }
                    else
                    {
                        Backward();
                    }
                    return "";
                }
                case "ceil":
                {
                    var result = this.Ceil();
                    return result.id + "";
                }
                case "ceil_":
                {
                    this.Ceil(inline: true);
                    return this.id + "";
                }
                case "copy":
                {
                    var result = Copy();
                    return result.Id.ToString();
                }
                case "cos":
                {
                    var result = Cos();
                    return result.Id.ToString();
                }
                case "cos_":
                {
                    Cos(inline: true);
                    return Id.ToString();
                }
                case "cosh":
                {
                    var result = Cosh();
                    return result.Id.ToString();
                }
                case "Cosh_":
                {
                    Cosh(inline: true);
                    return Id.ToString();
                }
                case "cpu":
                {
                    Cpu();
                    return msgObj.functionCall + ": OK";
                }
                case "div_elem":
                {
                    var tensor_1 = ctrl.getTensor(int.Parse(msgObj.tensorIndexParams[0]));
                    var result = this.Div(tensor_1);
                    return result.Id + "";
                }
                case "div_elem_":
                {
                    var tensor_1 = ctrl.getTensor(int.Parse(msgObj.tensorIndexParams[0]));
                    this.Div(tensor_1, inline: true);
                    return this.id + "";
                }
                case "div_scalar":
                {
                    FloatTensor result = Div(float.Parse(msgObj.tensorIndexParams[0]));

                    return result.Id + "";
                }
                case "div_scalar_":
                {
                    this.Div(float.Parse(msgObj.tensorIndexParams[0]), inline: true);
                    return this.id + "";
                }
                case "exp":
                {
                    var result = Exp();
                    return result.Id.ToString();
                }
                case "mul_scalar":
                {
                    FloatTensor result = Mul(float.Parse(msgObj.tensorIndexParams[0]));

                    return result.id + "";
                }
                case "mul_scalar_":
                {
                    this.Mul(float.Parse(msgObj.tensorIndexParams[0]), inline: true);
                    return this.id + "";
                }
                case "pow_elem":
                {
                    var tensor_1 = ctrl.getTensor(int.Parse(msgObj.tensorIndexParams[0]));
                    var result = this.Pow(tensor_1);
                    return result.id + "";
                }
                case "floor":
                {
                    var result = this.Floor();
                    return result.id + "";
                }
                case "floor_":
                {
                    this.Floor(inline: true);
                    return this.id + "";
                }
                case "get":
                {
                    var param_to_get = msgObj.tensorIndexParams[0];
                    switch (param_to_get)
                    {
                        case "autograd":
                        {
                            if (this.autograd)
                                return "1";
                            return "0";
                        }
                        case "children":
                        {
                            if (children != null)
                            {
                                string children_str = "";
                                foreach (KeyValuePair<int, int> entry in children)
                                {
                                    children_str += (entry.Key + ",");
                                }
                                return children_str;
                            }
                            else
                            {
                                return "";
                            }
                        }
                        case "creation_op":
                        {
                            if (creation_op != null)
                                return creation_op;
                            return "";
                        }
                        case "creators":
                        {
                            if (creators != null)
                            {
                                string creators_str = "";
                                foreach (FloatTensor entry in creators)
                                {
                                    creators_str += (entry.id + ",");
                                }
                                return creators_str;
                            }
                            else
                            {
                                return "";
                            }
                        }
                        case "data":
                        {
                          string out_str = "";

                          if (dataOnGpu) {
                            int[] temp_data = new int[size];
                            dataBuffer.GetData (temp_data);
                            for (int i = 0; i < size; i++) {
                              out_str += temp_data [i] + ",";
                            }
                          } else {
                            for (int i = 0; i < size; i++) {
                              out_str += data [i] + ",";
                            }
                          }

                          return out_str;
                        }
                        case "dataOnGpu":
                        {
                            if (dataOnGpu)
                            {
                                return "1";
                            }
                            return "0";
                        }
                        case "grad":
                        {
                            if (grad == null)
                                return "";
                            return grad.id + "";
                        }
                        case "id":
                        {
                            return this.id + "";
                        }
                        case "keepgrads":
                        {
                            if (this.keepgrads)
                                return "1";
                            return "0";
                        }
                        case "size":
                        {
                            return this.size + "";
                        }
                        case "shape":
                        {
                            string shape_str = "";
                            for (int i = 0; i < shape.Length; i++)
                            {
                                shape_str += (shape[i] + ",");
                            }
                            return shape_str;
                        }
                    }
                    return "param not found or not configured with a getter";
                }
                case "gpu":
                {
                    if (Gpu(ctrl.GetShader()))
                    {
                        return msgObj.functionCall + ": OK : Moved data to GPU.";
                    }
                    else
                    {
                        return msgObj.functionCall + ": FAILED : Did not move data.";
                    }
                }
                case "mul_elem":
                {
                    var tensor_1 = ctrl.getTensor(int.Parse(msgObj.tensorIndexParams[0]));
                    var result = this.Mul(tensor_1);

                    return result.id + "";
                }
                case "mul_elem_":
                {
                    var tensor_1 = ctrl.getTensor(int.Parse(msgObj.tensorIndexParams[0]));
                    this.Mul(tensor_1, inline: true);
                    return this.id + "";
                }
                case "mm":
                {
                  var tensor_1 = ctrl.getTensor (int.Parse (msgObj.tensorIndexParams [0]));
                  var result = this.MM (tensor_1);
                  return result.id + "";
                }
                case "pow_elem_":
                {
                    var tensor_1 = ctrl.getTensor(int.Parse(msgObj.tensorIndexParams[0]));
                    this.Pow(tensor_1, inline: true);
                    return this.id + "";
                }
                case "pow_scalar":
                {
                    FloatTensor result = Pow(float.Parse(msgObj.tensorIndexParams[0]));
                    return result.id + "";
                }
                case "pow_scalar_":
                {
                    this.Pow(float.Parse(msgObj.tensorIndexParams[0]), inline: true);
                    return this.id + "";
                }
                case "set":
                {
                    var param_to_set = msgObj.tensorIndexParams[0];
                    switch (param_to_set)
                    {
                        case "autograd":
                        {
                            if (msgObj.tensorIndexParams[1] == "1")
                            {
                                InitAutograd();
                                return "1";
                            }
                            else
                            {
                                autograd = false;
                                return "0";
                            }
                        }
                    }
                    return "setter not recognized";
                }
                case "sigmoid_":
                {
                    this.Sigmoid(inline: true);
                    return this.id + "";
                }
                case "sigmoid":
                {
                    var result = this.Sigmoid();
                    return result.id + "";
                }
                case "sub_elem":
                {
                    Debug.LogFormat("sub_elem");
                    var tensor_1 = ctrl.getTensor(int.Parse(msgObj.tensorIndexParams[0]));
                    var result = this.Sub(tensor_1);

                    return result.Id + "";
                }
                case "sub_elem_":
                {
                    Debug.LogFormat("sub_elem_");
                    var tensor_1 = ctrl.getTensor(int.Parse(msgObj.tensorIndexParams[0]));
                    this.Sub(tensor_1, inline: true);
                    return this.id + "";
                }
                case "neg":
                {
                    var result = Neg();
                    return result.Id.ToString();
                }
                case "rsqrt":
                {
                    var result = Rsqrt();
                    return result.Id.ToString();
                }
                case "print":
                {
                    bool dataOriginallyOnGpu = dataOnGpu;
                    if (dataOnGpu)
                    {
                        Cpu();
                    }

                    string data = this.Print();
                    Debug.LogFormat("<color=cyan>Print:</color> {0}", string.Join(",", this.Data));

                    if (dataOriginallyOnGpu)
                    {
                        Gpu(ctrl.GetShader());
                    }
                    return data;
                }
                case "sign":
                {
                    Debug.LogFormat("sign");
                    var result = this.Sign();
                    return result.Id + "";
                }
                case "sign_":
                {
                    Debug.LogFormat("sign_");
                    Sign(inline: true);
                    return Id.ToString();
                }
                case "sin":
                {
                    var result = Sin();
                    return result.Id.ToString();
                }
                case "sin_":
                {
                    Sin(inline: true);
                    return Id.ToString();
                }
                case "sqrt":
                {
                    var result = Sqrt();
                    return result.id.ToString();
                }
                case "size":
                {
                    var result = SizeTensor();
                    return result.id.ToString();
                }

                case "sub_scalar":
                {
                    Debug.LogFormat("sub_scalar");
                    FloatTensor result = Sub(float.Parse(msgObj.tensorIndexParams[0]));

                    return result.Id + "";
                }
                case "sub_scalar_":
                {
                    Debug.LogFormat("sub_scalar_");
                    this.Sub(float.Parse(msgObj.tensorIndexParams[0]), inline: true);
                    return this.id + "";
                }
                case "sum_dim":
                {
                    Debug.LogFormat("sum_dim");
                    FloatTensor result = this.Sum(int.Parse(msgObj.tensorIndexParams[0]));
                    return result.Id + "";
                }
                case "to_numpy":
                {
                    return string.Join(" ", data);
                }
                case "tan":
                {
                    var result = Tan();
                    return result.Id.ToString();
                }
                case "tan_":
                {
                    Tan(inline: true);
                    return Id.ToString();
                }
                case "tanh":
                {
                    var result = Tanh();
                    return result.Id.ToString();
                }
                case "sinh":
                {
                    var result = Sinh();
                    return result.Id.ToString();
                }
                case "sinh_":
                {
                    Sinh(inline: true);
                    return Id.ToString();
                }
                case "transpose":
                {
                    var result = Transpose();
                    return result.Id.ToString();
                }

                case "triu":
                {
                    var K = int.Parse(msgObj.tensorIndexParams[0]);
                    var result = Copy();
                    result.Triu_(K);
                    return result.Id.ToString();
                }
                case "triu_":
                {
                    var K = int.Parse(msgObj.tensorIndexParams[0]);
                    Triu_(K);
                    return Id.ToString();
                }

                case "trunc":
                {
                    var result = Trunc();
                    return result.Id.ToString();
                }

                case "view":
                {
                    int[] new_dims = new int[msgObj.tensorIndexParams.Length];
                    for (int i = 0; i < msgObj.tensorIndexParams.Length; i++)
                    {
                        new_dims[i] = int.Parse(msgObj.tensorIndexParams[i]);
                    }
                    var result = View(new_dims);
                    return result.Id.ToString();
                }

                case "view_":
                {
                    int[] new_dims = new int[msgObj.tensorIndexParams.Length];
                    for (int i = 0; i < msgObj.tensorIndexParams.Length; i++)
                    {
                        new_dims[i] = int.Parse(msgObj.tensorIndexParams[i]);
                    }
                    View(new_dims, inline: true);
                    return Id.ToString();
                }

                case "zero_":
                {
                    Zero_();
                    return msgObj.functionCall + ": OK";
                }
                case "is_contiguous":
                {
                    return Convert.ToString(IsContiguous());
                }
                case "squeeze":
                {
                    if (msgObj.tensorIndexParams.Length > 0)
                    {
                        int dim = int.Parse(msgObj.tensorIndexParams[0]);
                        var result = Squeeze(dim: dim);
                        ctrl.addTensor(result);
                        return result.Id.ToString();
                    }
                    else
                    {
                        var result = Squeeze();
                        ctrl.addTensor(result);
                        return result.Id.ToString();
                    }
                }
                case "sqeeze_":
                {
                    if (msgObj.tensorIndexParams.Length > 0)
                    {
                        int dim = int.Parse(msgObj.tensorIndexParams[0]);
                        Squeeze(dim: dim, inline: true);
                        return Id.ToString();
                    }
                    else
                    {
                        Squeeze(inline: true);
                        return Id.ToString();
                    }
                }
                default:
                    break;
            }
            return "SyftController.processMessage: Command not found.";
        }

        public string Print()
        {
            bool dataOriginallyOnGpu = dataOnGpu;
            ComputeShader _shader = this.shader;

            if (dataOnGpu)
            {
                Cpu();
            }

            string print = "";

            if (shape.Length > 3)
                print += "Only printing the last 3 dimesnions\n";
            int d3 = 1;
            if (shape.Length > 2)
                d3 = shape[shape.Length - 3];
            int d2 = 1;
            if (shape.Length > 1)
                d2 = shape[shape.Length - 2];
            int d1 = shape[shape.Length - 1];

            for (int k = 0; k < d3; k++)
            {
                for (int j = 0; j < d2; j++)
                {
                    for (int i = 0; i < d1; i++)
                    {
                        float f = data[i + j * d1 + k * d1 * d2];
						            print += f.ToString("0.0000") + ", ";

                    }
                    print += "\n";
                }
                print += "\n";
            }

            if (dataOriginallyOnGpu)
            {
                Gpu(_shader);
            }
            return print;
        }
    }
}