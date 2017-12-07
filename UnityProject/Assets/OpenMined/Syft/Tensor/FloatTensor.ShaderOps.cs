﻿using UnityEngine;

namespace OpenMined.Syft.Tensor
{

public partial class FloatTensor
{
private ComputeShader shader;

[SerializeField]
private static int AbsKernel;
[SerializeField]
private static int AbsKernel_;
[SerializeField]
private static int AcosKernel;
[SerializeField]
private static int AcosKernel_;
[SerializeField]
private static int AsinKernel;
[SerializeField]
private static int AsinKernel_;
[SerializeField]
private static int AtanKernel;
[SerializeField]
private static int AtanKernel_;
[SerializeField]
private static int AddScalarKernel_;
[SerializeField]
private static int AddElemKernel_;
[SerializeField]
private static int AddScalarKernel;
[SerializeField]
private static int AddElemKernel;
[SerializeField]
private static int AddMMKernel_;
[SerializeField]
private static int AddMVKernel_;
[SerializeField]
private static int CeilKernel;
[SerializeField]
private static int CeilKernel_;
[SerializeField]
private static int CopyBufferKernel;
[SerializeField]
private static int CosKernel;
[SerializeField]
private static int CosKernel_;
[SerializeField]
private static int CoshKernel;
[SerializeField]
private static int CoshKernel_;
[SerializeField]
private static int DivScalarKernel_;
[SerializeField]
private static int DivElemKernel_;
[SerializeField]
private static int DivScalarKernel;
[SerializeField]
private static int DivElemKernel;
[SerializeField]
private static int ExpKernel;
[SerializeField]
private static int FloorKernel_;
[SerializeField]
private static int FloorKernel;
[SerializeField]
private static int MulScalarKernel_;
[SerializeField]
private static int MulElemKernel_;
[SerializeField]
private static int MulScalarKernel;
[SerializeField]
private static int MulElemKernel;
[SerializeField]
private static int PowScalarKernel_;
[SerializeField]
private static int PowElemKernel_;
[SerializeField]
private static int PowScalarKernel;
[SerializeField]
private static int PowElemKernel;
[SerializeField]
private static int NegateKernel;
[SerializeField]
private static int RsqrtKernel;
[SerializeField]
private static int SigmoidKernel;
[SerializeField]
private static int SigmoidKernel_;
[SerializeField]
private static int SignKernel;
[SerializeField]
private static int SignKernel_;
[SerializeField]
private static int SinKernel;
[SerializeField]
private static int SinKernel_;
[SerializeField]
private static int SqrtKernel;
[SerializeField]
private static int SubScalarKernel_;
[SerializeField]
private static int SubElemKernel_;
[SerializeField]
private static int SubScalarKernel;
[SerializeField]
private static int SubElemKernel;
[SerializeField]
private static int TanKernel;
[SerializeField]
private static int TanKernel_;
[SerializeField]
private static int TanhKernel;
[SerializeField]
private static int SinhKernel;
[SerializeField]
private static int SinhKernel_;
[SerializeField]
private static int TriuKernel_;
[SerializeField]
private static int TruncKernel;
[SerializeField]
private static int ZeroKernel_;

public void initShaderKernels ()
{
	if (shader != null) {
		// save shaders and kernels
		AbsKernel = shader.FindKernel ("Abs");
		AbsKernel_ = shader.FindKernel ("Abs_");
		AcosKernel = shader.FindKernel ("Acos");
		AcosKernel_ = shader.FindKernel ("Acos_");
		AsinKernel = shader.FindKernel ("Asin");
		AsinKernel_ = shader.FindKernel ("Asin_");
		AtanKernel = shader.FindKernel ("Atan");
		AtanKernel_ = shader.FindKernel ("Atan_");
		AddScalarKernel_ = shader.FindKernel ("AddScalar_");
		AddElemKernel_ = shader.FindKernel ("AddElem_");
		AddScalarKernel = shader.FindKernel ("AddScalar");
		AddElemKernel = shader.FindKernel ("AddElem");
		AddMMKernel_ = shader.FindKernel ("AddMM_");
		AddMVKernel_ = shader.FindKernel ("AddMV_");
		CeilKernel = shader.FindKernel ("Ceil");
		CeilKernel_ = shader.FindKernel ("Ceil_");
		CopyBufferKernel = shader.FindKernel ("CopyBuffer");
		CosKernel = shader.FindKernel ("Cos");
		CosKernel_ = shader.FindKernel ("Cos_");
		CoshKernel = shader.FindKernel ("Cosh");
		CoshKernel_ = shader.FindKernel ("Cosh_");
		DivScalarKernel_ = shader.FindKernel ("DivScalar_");
		DivElemKernel_ = shader.FindKernel ("DivElem_");
		DivScalarKernel = shader.FindKernel ("DivScalar");
		DivElemKernel = shader.FindKernel ("DivElem");
		ExpKernel = shader.FindKernel ("Exp");
		FloorKernel_ = shader.FindKernel ("Floor_");
		FloorKernel = shader.FindKernel ("Floor");
		MulScalarKernel_ = shader.FindKernel ("MulScalar_");
		MulElemKernel_ = shader.FindKernel ("MulElem_");
		MulScalarKernel = shader.FindKernel ("MulScalar");
		MulElemKernel = shader.FindKernel ("MulElem");
		PowScalarKernel_ = shader.FindKernel ("PowScalar_");
		PowElemKernel_ = shader.FindKernel ("PowElem_");
		PowScalarKernel = shader.FindKernel ("PowScalar");
		PowElemKernel = shader.FindKernel ("PowElem");
		NegateKernel = shader.FindKernel ("Negate");
		RsqrtKernel = shader.FindKernel ("Rsqrt");
		// PowKernel = shader.FindKernel ("Pow");
		// PowKernel_ = shader.FindKernel ("Pow_");
		SigmoidKernel = shader.FindKernel ("Sigmoid");
		SigmoidKernel_ = shader.FindKernel ("Sigmoid_");
		SignKernel = shader.FindKernel ("Sign");
		SignKernel_ = shader.FindKernel("Sign_");
		SinKernel = shader.FindKernel ("Sin");
		SinKernel_ = shader.FindKernel ("Sin_");
		SqrtKernel = shader.FindKernel ("Sqrt");
		SubScalarKernel_ = shader.FindKernel ("SubScalar_");
		SubElemKernel_ = shader.FindKernel ("SubElem_");
		SubScalarKernel = shader.FindKernel ("SubScalar");
		SubElemKernel = shader.FindKernel ("SubElem");
		TanKernel = shader.FindKernel ("Tan");
		TanKernel_ = shader.FindKernel ("Tan_");
		TanhKernel = shader.FindKernel ("Tanh");
		SinhKernel = shader.FindKernel ("Sinh");
		SinhKernel_ = shader.FindKernel ("Sinh_");
		TriuKernel_ = shader.FindKernel ("Triu_");
		TruncKernel = shader.FindKernel ("Trunc");
		ZeroKernel_ = shader.FindKernel ("Zero_");
	}

}

public FloatTensor AbsGPU (FloatTensor result)
{
	Debug.LogFormat ("<color=blue>FloatTensor.AbsGPU dataOnGpu: {0}</color>", dataOnGpu);
	if (dataOnGpu) {
		shader.SetBuffer (AbsKernel, "AbsData", dataBuffer);
		shader.SetBuffer (AbsKernel, "AbsResult", result.dataBuffer);
		shader.Dispatch (AbsKernel, this.size, 1, 1);
	}
	return result;
}

public void AbsGPU_ ()
{
	Debug.LogFormat ("<color=blue>FloatTensor.AbsGPU_ dataOnGpu: {0}</color>", dataOnGpu);
	if (dataOnGpu) {
		shader.SetBuffer (AbsKernel_, "AbsData_", dataBuffer);
		shader.Dispatch (AbsKernel_, this.size, 1, 1);
	}
}

public FloatTensor AcosGPU ()
{
	var result = new FloatTensor (ctrl, _shape:shape, _shader:this.shader, _dataOnGpu:dataOnGpu);
	shader.SetBuffer (AcosKernel, "AcosData", dataBuffer);
	shader.SetBuffer (AcosKernel, "AcosResult", result.DataBuffer);
	shader.Dispatch (AcosKernel, this.size, 1, 1);
	return result;
}

public void AcosGPU_ ()
{
	shader.SetBuffer (AcosKernel_, "AcosData_", dataBuffer);
	shader.Dispatch (AcosKernel, this.size, 1, 1);
}

public FloatTensor AsinGPU ()
{
	var result = this.emptyTensorCopy();
	shader.SetBuffer (AsinKernel, "AsinData", dataBuffer);
	shader.SetBuffer (AsinKernel, "AsinResult", result.DataBuffer);
	shader.Dispatch (AsinKernel, this.size, 1, 1);
	return result;
}

public void AsinGPU_ ()
{
	shader.SetBuffer (AsinKernel_, "AsinData_", dataBuffer);
	shader.Dispatch (AsinKernel, this.size, 1, 1);
}

public FloatTensor AtanGPU ()
{
	var result = this.emptyTensorCopy();
	shader.SetBuffer (AtanKernel, "AtanData", dataBuffer);
	shader.SetBuffer (AtanKernel, "AtanResult", result.DataBuffer);
	shader.Dispatch (AtanKernel, this.size, 1, 1);
	return result;
}

public void AtanGPU_ ()
{
	shader.SetBuffer (AtanKernel_, "AtanData_", dataBuffer);
	shader.Dispatch (AtanKernel, this.size, 1, 1);
}

public void AddScalarGPU_ (float value)
{
	Debug.LogFormat ("<color=blue>FloatTensor.AddScalarGPU_ dataOnGpu: {0}</color>", dataOnGpu);

	if (dataOnGpu) {
		var valBuffer = SendFloatToGpu (AddScalarKernel_, value, "AddScalarScalar_");

		shader.SetBuffer (AddScalarKernel_, "AddScalarData_", dataBuffer);
		shader.Dispatch (AddScalarKernel_, this.size, 1, 1);

		valBuffer.Release ();
	}
}

public void AddElemGPU_ (FloatTensor tensor)
{
	Debug.LogFormat ("<color=blue>FloatTensor.AddElemGPU_ dataOnGpu: {0}</color>", dataOnGpu);

	if (dataOnGpu) {
		if (this.id != tensor.id) {
			shader.SetBuffer (AddElemKernel_, "AddElemDataA_", dataBuffer);
			shader.SetBuffer (AddElemKernel_, "AddElemDataB_", tensor.dataBuffer);
			shader.Dispatch (AddElemKernel_, this.size, 1, 1);
		} else {
			Debug.LogFormat ("addition with itself should be multiplication instead", dataOnGpu);
			this.MulScalarGPU_ (2);
		}

	}
}

public FloatTensor AddScalarGPU (float value, FloatTensor result)
{
	Debug.LogFormat ("<color=blue>FloatTensor.AddScalarGPU dataOnGpu: {0}</color>", dataOnGpu);

	if (dataOnGpu) {
		var valBuffer = SendFloatToGpu (AddScalarKernel, value, "AddScalarScalar");

		shader.SetBuffer (AddScalarKernel, "AddScalarData", dataBuffer);
		shader.SetBuffer (AddScalarKernel, "AddScalarResult", result.dataBuffer);
		shader.Dispatch (AddScalarKernel, this.size, 1, 1);

		valBuffer.Release ();
	}
	return result;
}

public FloatTensor AddElemGPU (FloatTensor tensor, FloatTensor result)
{

	Debug.LogFormat ("<color=blue>FloatTensor.AddElemGPU dataOnGpu: {0}</color>", dataOnGpu);

	if (dataOnGpu) {
		// if tensors are not actually referring to the same tensor
		if (this.id != tensor.id) {
			shader.SetBuffer (AddElemKernel, "AddElemDataA", this.DataBuffer);
			shader.SetBuffer (AddElemKernel, "AddElemDataB", tensor.DataBuffer);
			shader.SetBuffer (AddElemKernel, "AddElemDataResult", result.DataBuffer);
			shader.Dispatch (AddElemKernel, this.size, 1, 1);
		} else {
			Debug.LogFormat ("addition with itself should be multiplication instead", dataOnGpu);
			return this.MulScalarGPU (2, result);
		}


	}
	return result;
}

public void CopyBuffer (ComputeBuffer buff1, ComputeBuffer buff2)
{
	shader.SetBuffer (CopyBufferKernel, "buffer1", buff1);
	shader.SetBuffer (CopyBufferKernel, "buffer2", buff2);
	shader.Dispatch (CopyBufferKernel, this.size, 1, 1);
}
public FloatTensor CosGPU ()
{
	var result = this.emptyTensorCopy();
	shader.SetBuffer (CosKernel, "CosData", dataBuffer);
	shader.SetBuffer (CosKernel, "CosResult", result.DataBuffer);
	shader.Dispatch (CosKernel, this.size, 1, 1);
	return result;
}

public void CosGPU_ ()
{
	shader.SetBuffer (CosKernel_, "CosData_", dataBuffer);
	shader.Dispatch (CosKernel, this.size, 1, 1);
}

public FloatTensor CoshGPU ()
{
	var result = this.emptyTensorCopy();
	shader.SetBuffer (CoshKernel, "CoshData", dataBuffer);
	shader.SetBuffer (CoshKernel, "CoshResult", result.DataBuffer);
	shader.Dispatch (CoshKernel, this.size, 1, 1);
	return result;
}

public void CoshGPU_ ()
{
	shader.SetBuffer (CoshKernel_, "CoshData_", dataBuffer);
	shader.Dispatch (CoshKernel, this.size, 1, 1);
}

public void DivScalarGPU_ (float value)
{
	Debug.LogFormat ("<color=blue>FloatTensor.DivScalarGPU_ dataOnGpu: {0}</color>", dataOnGpu);

	if (dataOnGpu) {
		var valBuffer = SendFloatToGpu (DivScalarKernel_, value, "DivScalarScalar_");

		shader.SetBuffer (DivScalarKernel_, "DivScalarData_", dataBuffer);
		shader.Dispatch (DivScalarKernel_, this.size, 1, 1);

		valBuffer.Release ();
	}
}

public void DivElemGPU_ (FloatTensor tensor)
{
	Debug.LogFormat ("<color=blue>FloatTensor.DivElemGPU_ dataOnGpu: {0}</color>", dataOnGpu);

	if (dataOnGpu) {
		if (tensor.id != this.id) {
			shader.SetBuffer (DivElemKernel_, "DivElemDataA_", dataBuffer);
			shader.SetBuffer (DivElemKernel_, "DivElemDataB_", tensor.dataBuffer);
			shader.Dispatch (DivElemKernel_, this.size, 1, 1);
		} else {
			tensor.Zero_();
			tensor.Add(1, inline: true);
		}
	}
}

public FloatTensor DivScalarGPU (float value, FloatTensor result)
{
	Debug.LogFormat ("<color=blue>FloatTensor.DivScalarGPU dataOnGpu: {0}</color>", dataOnGpu);

	if (dataOnGpu) {
		var valBuffer = SendFloatToGpu (DivScalarKernel, value, "DivScalarScalar");

		shader.SetBuffer (DivScalarKernel, "DivScalarData", dataBuffer);
		shader.SetBuffer (DivScalarKernel, "DivScalarResult", result.dataBuffer);
		shader.Dispatch (DivScalarKernel, this.size, 1, 1);

		valBuffer.Release ();
	}
	return result;
}

public FloatTensor DivElemGPU (FloatTensor tensor, FloatTensor result)
{
	Debug.LogFormat ("<color=blue>FloatTensor.DivElemGPU dataOnGpu: {0}</color>", dataOnGpu);

	if (dataOnGpu) {
		if (tensor.id != this.id) {
			shader.SetBuffer (DivElemKernel, "DivElemDataA", dataBuffer);
			shader.SetBuffer (DivElemKernel, "DivElemDataB", tensor.dataBuffer);
			shader.SetBuffer (DivElemKernel, "DivElemDataResult", result.dataBuffer);
			shader.Dispatch (DivElemKernel, this.size, 1, 1);
		} else {
			result.Add (1, inline: true);
			return result;
		}

	}
	return result;
}


public void AddMatrixMultiplyGPU (FloatTensor tensor_1, FloatTensor tensor_2)
{
	//Debug.LogFormat("<color=blue>FloatTensor.add_matrix_multiply dataOnGpu: {0}</color>", dataOnGpu);
	shader.SetBuffer (AddMMKernel_, "AddmmDataA", dataBuffer);
	shader.SetBuffer (AddMMKernel_, "AddmmDataB", tensor_1.DataBuffer);                 //d
	shader.SetBuffer (AddMMKernel_, "AddmmDataC", tensor_2.DataBuffer);
	shader.Dispatch (AddMMKernel_, size, 1, 1);
}

public void InitAddMatrixMultiplyGpu (FloatTensor tensor_1)
{
	var dim = new Dimensions[] {
		new Dimensions (tensor_1.shape.Length, tensor_1.shape [0])
	};

	var dimBuffer = new ComputeBuffer (dim.Length, dim [0].Stride ());
	dimBuffer.SetData (dim);
	shader.SetBuffer (AddMMKernel_, "AddmmDimensions", dimBuffer);
}

public FloatTensor CeilGPU(FloatTensor result)
{
	if (!dataOnGpu) return this;
	shader.SetBuffer(CeilKernel, "CeilData", dataBuffer);
	shader.SetBuffer(CeilKernel, "CeilResult", result.DataBuffer);
	shader.Dispatch(CeilKernel, 1, 1, 1);
	return result;
}

public void AddMatrixVectorProductGPU (FloatTensor matrix, FloatTensor vector)
{
	var refShapeBuffer = SendIntToGpu (AddMVKernel_, this.Shape [0], "AddMVRefShape_");
	shader.SetBuffer (AddMVKernel_, "AddMVRefData", dataBuffer);
	shader.SetBuffer (AddMVKernel_, "AddMVMatrixData", matrix.DataBuffer);                 //d
	shader.SetBuffer (AddMVKernel_, "AddMVVectorData", vector.DataBuffer);
	shader.Dispatch (AddMVKernel_, this.Size, 1, 1);
	refShapeBuffer.Release ();
}


public void CeilGPU_()
{
	Debug.LogFormat("<color=blue>FloatTensor.ceil_ dataOnGpu: {0}</color>", dataOnGpu);

	shader.SetBuffer(CeilKernel_, "CeilData_", dataBuffer);
	shader.Dispatch(CeilKernel_, 1, 1, 1);
}

private FloatTensor ExpGPU()
{
	if (!dataOnGpu) return this;

	var result = new FloatTensor (_ctrl: null, _shape:shape, _shader:this.shader, _dataOnGpu:dataOnGpu);
	shader.SetBuffer(SqrtKernel, "ExptData", dataBuffer);
	shader.SetBuffer(SqrtKernel, "ExpResult", result.dataBuffer);
	shader.Dispatch(SqrtKernel, size, 1, 1);

	return result;
}

public void FloorGPU_()
{
	Debug.LogFormat("<color=blue>FloatTensor.floor_ dataOnGpu: {0}</color>", dataOnGpu);

	shader.SetBuffer(FloorKernel_, "FloorData_", dataBuffer);
	shader.Dispatch(FloorKernel_, 1, 1, 1);
}

public FloatTensor FloorGPU(FloatTensor result)
{
	Debug.LogFormat("<color=blue>FloatTensor.floor dataOnGpu: {0}</color>", dataOnGpu);

	if (result.DataOnGpu)
	{
		shader.SetBuffer(FloorKernel, "FloorData", dataBuffer);
		shader.SetBuffer(FloorKernel, "FloorResult", result.dataBuffer);
		shader.Dispatch(FloorKernel, 1, 1, 1);
	}
	return result;
}

public void MulScalarGPU_ (float value)
{
	Debug.LogFormat ("<color=blue>FloatTensor.MulScalarGPU_ dataOnGpu: {0}</color>", dataOnGpu);

	if (dataOnGpu) {
		var valBuffer = SendFloatToGpu (MulScalarKernel_, value, "MulScalarScalar_");

		shader.SetBuffer (MulScalarKernel_, "MulScalarData_", dataBuffer);
		shader.Dispatch (MulScalarKernel_, this.size, 1, 1);

		valBuffer.Release ();
	}
}

public void MulElemGPU_ (FloatTensor tensor)
{
	Debug.LogFormat ("<color=blue>FloatTensor.MulElemGPU_ dataOnGpu: {0}</color>", dataOnGpu);

	if (dataOnGpu) {
		if (tensor.id != this.id) {
			shader.SetBuffer (MulElemKernel_, "MulElemDataA_", dataBuffer);
			shader.SetBuffer (MulElemKernel_, "MulElemDataB_", tensor.dataBuffer);
			shader.Dispatch (MulElemKernel_, this.size, 1, 1);
		} else {
			PowScalarGPU_ (2);
		}

	}
}

public FloatTensor MulScalarGPU (float value, FloatTensor result)
{
	Debug.LogFormat ("<color=blue>FloatTensor.MulScalarGPU dataOnGpu: {0}</color>", dataOnGpu);

	if (dataOnGpu) {
		var valBuffer = SendFloatToGpu (MulScalarKernel, value, "MulScalarScalar");

		shader.SetBuffer (MulScalarKernel, "MulScalarData", dataBuffer);
		shader.SetBuffer (MulScalarKernel, "MulScalarResult", result.dataBuffer);
		shader.Dispatch (MulScalarKernel, this.size, 1, 1);

		valBuffer.Release ();
	}
	return result;
}

public FloatTensor MulElemGPU (FloatTensor tensor, FloatTensor result)
{
	Debug.LogFormat ("<color=blue>FloatTensor.MulElemGPU dataOnGpu: {0}</color>", dataOnGpu);

	if (dataOnGpu) {
		if (tensor.id != this.id) {
			shader.SetBuffer (MulElemKernel, "MulElemDataA", dataBuffer);
			shader.SetBuffer (MulElemKernel, "MulElemDataB", tensor.dataBuffer);
			shader.SetBuffer (MulElemKernel, "MulElemDataResult", result.dataBuffer);
			shader.Dispatch (MulElemKernel, this.size, 1, 1);
		} else {
			return this.PowScalarGPU (2, result);
		}

	}
	return result;
}

public void PowScalarGPU_ (float value)
{
	Debug.LogFormat ("<color=blue>FloatTensor.PowScalarGPU_ dataOnGpu: {0}</color>", dataOnGpu);

	if (dataOnGpu) {
		var valBuffer = SendFloatToGpu (PowScalarKernel_, value, "PowScalarScalar_");

		shader.SetBuffer (PowScalarKernel_, "PowScalarData_", dataBuffer);
		shader.Dispatch (PowScalarKernel_, this.size, 1, 1);

		valBuffer.Release ();
	}
}

public void PowElemGPU_ (FloatTensor tensor)
{
	Debug.LogFormat ("<color=blue>FloatTensor.PowElemGPU_ dataOnGpu: {0}</color>", dataOnGpu);

	if (dataOnGpu) {
		shader.SetBuffer (PowElemKernel_, "PowElemDataA_", dataBuffer);
		shader.SetBuffer (PowElemKernel_, "PowElemDataB_", tensor.dataBuffer);
		shader.Dispatch (PowElemKernel_, this.size, 1, 1);
	}
}

public FloatTensor PowScalarGPU (float value, FloatTensor result)
{
	Debug.LogFormat ("<color=blue>FloatTensor.PowScalarGPU dataOnGpu: {0}</color>", dataOnGpu);

	if (dataOnGpu) {
		var valBuffer = SendFloatToGpu (PowScalarKernel, value, "PowScalarScalar");

		shader.SetBuffer (PowScalarKernel, "PowScalarData", dataBuffer);
		shader.SetBuffer (PowScalarKernel, "PowScalarResult", result.dataBuffer);
		shader.Dispatch (PowScalarKernel, this.size, 1, 1);
		valBuffer.Release ();
	}
	return result;
}


public FloatTensor PowElemGPU (FloatTensor tensor, FloatTensor result)
{
	Debug.LogFormat ("<color=blue>FloatTensor.PowElemGPU dataOnGpu: {0}</color>", dataOnGpu);

	if (dataOnGpu) {
		shader.SetBuffer (PowElemKernel, "PowElemDataA", dataBuffer);
		shader.SetBuffer (PowElemKernel, "PowElemDataB", tensor.dataBuffer);
		shader.SetBuffer (PowElemKernel, "PowElemDataResult", result.dataBuffer);
		shader.Dispatch (PowElemKernel, this.size, 1, 1);

	}
	return result;
}


public FloatTensor NegateGPU ()
{
	if (dataOnGpu) {
		var result = new FloatTensor (_ctrl: null, _shape:shape, _shader:this.shader, _dataOnGpu:dataOnGpu);
		shader.SetBuffer (NegateKernel, "NegateData", dataBuffer);
		shader.SetBuffer (NegateKernel, "NegateResult", result.dataBuffer);
		shader.Dispatch (NegateKernel, 1, 1, 1);
		return result;
	}

	return this;
}

public FloatTensor RsqrtGPU()
{
	if (dataOnGpu)
	{
		var result = new FloatTensor (_ctrl: null, _shape:shape, _shader:this.shader, _dataOnGpu:dataOnGpu);
		shader.SetBuffer(RsqrtKernel, "RsqrtData", dataBuffer);
		shader.SetBuffer(RsqrtKernel, "RsqrtResult", result.dataBuffer);
		shader.Dispatch(RsqrtKernel, 1, 1, 1);
		return result;
	}

	return this;
}

// public FloatTensor PowGPU(float value, FloatTensor result)
// {
//      Debug.LogFormat("<color=blue>FloatTensor.PowGPU dataOnGpu: {0}</color>", dataOnGpu);
//
//      if (dataOnGpu)
//      {
//              var valBuffer = SendFloatToGpu(PowKernel, value, "PowScalarScalar");
//
//              shader.SetBuffer(PowKernel, "PowScalarData", dataBuffer);
//              shader.SetBuffer(PowKernel, "PowScalarResult", result.dataBuffer);
//              shader.Dispatch(PowKernel, this.size, 1, 1);
//
//              valBuffer.Release();
//      }
//      return result;
// }
//
// public void PowGPU_(float value)
// {
//      Debug.LogFormat("<color=blue>FloatTensor.PowGPU_ dataOnGpu: {0}</color>", dataOnGpu);
//
//      if (dataOnGpu)
//      {
//              var valBuffer = SendFloatToGpu(PowKernel_, value, "PowScalarScalar_");
//
//              shader.SetBuffer(PowKernel_, "PowScalarData_", dataBuffer);
//              shader.Dispatch(PowKernel_, this.size, 1, 1);
//
//              valBuffer.Release();
//      }
// }

private FloatTensor SqrtGPU()
{
	if (!dataOnGpu) return this;

	var result = new FloatTensor (_ctrl: null, _shape:shape, _shader:this.shader, _dataOnGpu:dataOnGpu);
	shader.SetBuffer(SqrtKernel, "SqrtData", dataBuffer);
	shader.SetBuffer(SqrtKernel, "SqrtResult", result.dataBuffer);
	shader.Dispatch(SqrtKernel, size, 1, 1);

	return result;
}

public void SigmoidGPU_()
{
	Debug.LogFormat("<color=blue>FloatTensor.SigmoidGPU_ dataOnGpu: {0}</color>", dataOnGpu);
	shader.SetBuffer(SigmoidKernel_, "SigmoidData_", dataBuffer);
	shader.Dispatch(SigmoidKernel_, this.size, 1, 1);
}

public FloatTensor SigmoidGPU(FloatTensor result)
{
	Debug.LogFormat("<color=blue>FloatTensor.SigmoidGPU dataOnGpu: {0}</color>", dataOnGpu);
	shader.SetBuffer(SigmoidKernel, "SigmoidData", this.dataBuffer);
	shader.SetBuffer(SigmoidKernel, "SigmoidResult", result.dataBuffer);
	shader.Dispatch(SigmoidKernel, this.size, 1, 1);
	return result;
}

public FloatTensor SignGPU (FloatTensor result)
{
	Debug.LogFormat ("<color=blue>FloatTensor.SignGPU dataOnGpu: {0}</color>", dataOnGpu);
	if (dataOnGpu) {
		shader.SetBuffer (SignKernel, "SignData", dataBuffer);
		shader.SetBuffer (SignKernel, "SignResult", result.dataBuffer);
		shader.Dispatch (SignKernel, this.size, 1, 1);
	}
	return result;
}

public void SignGPU_() {
	Debug.LogFormat("<color=blue>FloatTensor.SignGPU_ dataOnGpu: {0}</color>", dataOnGpu);
	if (dataOnGpu) {
		shader.SetBuffer (SignKernel_, "SignData_", dataBuffer);
		shader.Dispatch (SignKernel_, this.size, 1, 1);
	}
}

public FloatTensor SinGPU ()
{
	var result = this.emptyTensorCopy();
	shader.SetBuffer (SinKernel, "SinData", dataBuffer);
	shader.SetBuffer (SinKernel, "SinResult", result.DataBuffer);
	shader.Dispatch (SinKernel, this.size, 1, 1);
	return result;
}

public void SinGPU_ ()
{
	shader.SetBuffer (SinKernel_, "SinData_", dataBuffer);
	shader.Dispatch (SinKernel, this.size, 1, 1);
}


public void SubScalarGPU_ (float value)
{
	Debug.LogFormat ("<color=blue>FloatTensor.SubScalarGPU_ dataOnGpu: {0}</color>", dataOnGpu);

	if (dataOnGpu) {
		var valBuffer = SendFloatToGpu (SubScalarKernel_, value, "SubScalarScalar_");

		shader.SetBuffer (SubScalarKernel_, "SubScalarData_", dataBuffer);
		shader.Dispatch (SubScalarKernel_, this.size, 1, 1);

		valBuffer.Release ();
	}
}

public void SubElemGPU_ (FloatTensor tensor)
{
	Debug.LogFormat ("<color=blue>FloatTensor.SubElemGPU_ dataOnGpu: {0}</color>", dataOnGpu);

	if (dataOnGpu) {
		if (this.id != tensor.id) {
			shader.SetBuffer (SubElemKernel_, "SubElemDataA_", dataBuffer);
			shader.SetBuffer (SubElemKernel_, "SubElemDataB_", tensor.dataBuffer);
			shader.Dispatch (SubElemKernel_, this.size, 1, 1);
		} else {
			Debug.LogFormat ("addition with itself should be multiplication instead", dataOnGpu);
			this.Zero_ ();
		}

	}
}

public FloatTensor SubScalarGPU (float value, FloatTensor result)
{
	Debug.LogFormat ("<color=blue>FloatTensor.SubScalarGPU dataOnGpu: {0}</color>", dataOnGpu);

	if (dataOnGpu) {
		var valBuffer = SendFloatToGpu (SubScalarKernel, value, "SubScalarScalar");

		shader.SetBuffer (SubScalarKernel, "SubScalarData", dataBuffer);
		shader.SetBuffer (SubScalarKernel, "SubScalarResult", result.dataBuffer);
		shader.Dispatch (SubScalarKernel, this.size, 1, 1);

		valBuffer.Release ();
	}
	return result;
}

public FloatTensor SubElemGPU (FloatTensor tensor, FloatTensor result)
{

	Debug.LogFormat ("<color=blue>FloatTensor.SubElemGPU dataOnGpu: {0}</color>", dataOnGpu);

	if (dataOnGpu) {
		// if tensors are not actually referring to the same tensor
		if (this.id != tensor.id) {
			shader.SetBuffer (SubElemKernel, "SubElemDataA", this.DataBuffer);
			shader.SetBuffer (SubElemKernel, "SubElemDataB", tensor.DataBuffer);
			shader.SetBuffer (SubElemKernel, "SubElemDataResult", result.DataBuffer);
			shader.Dispatch (SubElemKernel, this.size, 1, 1);
		} else {
			// should return a tensor of zeros.
			return result;
		}


	}
	return result;
}

public FloatTensor TanGPU ()
{
	var result = this.emptyTensorCopy();
	shader.SetBuffer (TanKernel, "TanData", dataBuffer);
	shader.SetBuffer (TanKernel, "TanResult", result.DataBuffer);
	shader.Dispatch (TanKernel, this.size, 1, 1);
	return result;
}

public void TanGPU_ ()
{
	shader.SetBuffer (TanKernel_, "TanData_", dataBuffer);
	shader.Dispatch (TanKernel, this.size, 1, 1);
}


public FloatTensor TanhGPU ()
{
	var result = new FloatTensor (_ctrl: ctrl, _shape:shape, _shader:this.shader, _dataOnGpu:dataOnGpu);
	shader.SetBuffer (TanhKernel, "TanhData", dataBuffer);
	shader.SetBuffer (TanhKernel, "TanhResult", result.DataBuffer);
	shader.Dispatch (TanhKernel, this.size, 1, 1);
	return result;
}

public FloatTensor SinhGPU ()
{
	var result = this.emptyTensorCopy();
	shader.SetBuffer (SinhKernel, "SinhData", dataBuffer);
	shader.SetBuffer (SinhKernel, "SinhResult", result.DataBuffer);
	shader.Dispatch (SinhKernel, this.size, 1, 1);
	return result;
}

public void SinhGPU_ ()
{
	shader.SetBuffer (SinhKernel_, "SinhData_", dataBuffer);
	shader.Dispatch (SinhKernel, this.size, 1, 1);
}

public void TriuGPU_ (int k)
{
	var dim = new Dimensions[] {
		new Dimensions (this.shape [0], this.shape [1])
	};

	var dimBuffer = new ComputeBuffer (2, dim [0].Stride ());
	var kBuffer = SendIntToGpu (TriuKernel_, k, "TriuK_");
	dimBuffer.SetData (dim);
	shader.SetBuffer (TriuKernel_, "TriuDimensions_", dimBuffer);
	shader.SetBuffer (TriuKernel_, "TriuData_", dataBuffer);
	shader.Dispatch (TriuKernel_, this.size, 1, 1);

	dimBuffer.Release ();
	kBuffer.Release ();

}

public FloatTensor TruncGPU ()
{
	var result = new FloatTensor (_ctrl: ctrl, _shape:shape, _shader:this.shader, _dataOnGpu:dataOnGpu);
	shader.SetBuffer (TruncKernel, "TruncData", dataBuffer);
	shader.SetBuffer (TruncKernel, "TruncResult", result.DataBuffer);
	shader.Dispatch (TruncKernel, this.size, 1, 1);
	return result;
}


public void ZeroGPU_ ()
{
	shader.SetBuffer (ZeroKernel_, "ZeroData_", dataBuffer);
	shader.Dispatch (ZeroKernel_, this.size, 1, 1);
}

public struct Dimensions
{
	public int rows, columns;

	public Dimensions (int _rows, int _columns)
	{
		rows = _rows;
		columns = _columns;
	}

	public int Stride ()
	{
		return 2 * sizeof(int);
	}
}

private ComputeBuffer SendFloatToGpu (int kernel, float value, string name)
{
	float[] scalarArray = new float[1];
	scalarArray [0] = value;

	var scalarBuffer = new ComputeBuffer (1, sizeof(float));
	scalarBuffer.SetData (scalarArray);
	shader.SetBuffer (kernel, name, scalarBuffer);

	return scalarBuffer;
}

private ComputeBuffer SendIntToGpu (int kernel, int value, string name)
{
	int[] array = new int[1];
	array [0] = value;

	var arrayBuffer = new ComputeBuffer (1, sizeof(float));
	arrayBuffer.SetData (array);
	shader.SetBuffer (kernel, name, arrayBuffer);

	return arrayBuffer;
}
}
}
