// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace ck_tile {

// This class is used for codegen pattern matching
enum class BlockFmhaPipelineEnum
{
    QRKSVS = 0,
    QRKSVS_ASYNC,
    QSKSVS,
    QRKSVS_ASYNC_TRLOAD,
    QRKSVS_ASYNC_TRLOAD_V3,
    QRKSVS_HDIM_VEC8,
};

template <BlockFmhaPipelineEnum>
struct BlockFmhaPipelineEnumToStr;

template <>
struct BlockFmhaPipelineEnumToStr<BlockFmhaPipelineEnum::QRKSVS>
{
    static constexpr const char* name = "qr";
};
template <>
struct BlockFmhaPipelineEnumToStr<BlockFmhaPipelineEnum::QRKSVS_ASYNC>
{
    static constexpr const char* name = "qr_async";
};
template <>
struct BlockFmhaPipelineEnumToStr<BlockFmhaPipelineEnum::QSKSVS>
{
    static constexpr const char* name = "qs";
};

template <>
struct BlockFmhaPipelineEnumToStr<BlockFmhaPipelineEnum::QRKSVS_ASYNC_TRLOAD>
{
    static constexpr const char* name = "qr_async_trload";
};

template <>
struct BlockFmhaPipelineEnumToStr<BlockFmhaPipelineEnum::QRKSVS_HDIM_VEC8>
{
    static constexpr const char* name = "qr_hdim_vec8";
};

} // namespace ck_tile
