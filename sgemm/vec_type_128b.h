#pragma once

template<typename T>
struct VecType128b;

template<>
struct VecType128b<float> {
    using Vec = float4;
};