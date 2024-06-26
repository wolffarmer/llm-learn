/* Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.
 * You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include "register/register.h"

namespace domi {
Status ParseParamsMatmulInteger(const ge::Operator& op_src, ge::Operator& op_dest) {
  return SUCCESS;
}

REGISTER_CUSTOM_OP("BatchMatMulV2")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::14::MatMulInteger"),
                   ge::AscendString("ai.onnx::15::MatMulInteger"),
                   ge::AscendString("ai.onnx::10::MatMulInteger"),
                   ge::AscendString("ai.onnx::11::MatMulInteger"),
                   ge::AscendString("ai.onnx::12::MatMulInteger"),
                   ge::AscendString("ai.onnx::13::MatMulInteger")})
    .ParseParamsByOperatorFn(ParseParamsMatmulInteger)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
