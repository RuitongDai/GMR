#!/usr/bin/env python3
"""
裁剪 BVH 动作文件，只保留指定的连续帧区间。

帧编号采用从 0 开始的方式，并且 end_frame 包含在保留范围内。

示例：
    python scripts/clip_bvh.py --input_file /home/dai/data/fallAndGetUp1_subject1.bvh --output_file /home/dai/data/getup1.bvh --start_frame 2070 --end_frame 2200
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="裁剪 BVH 文件，只保留指定的连续动作帧。"
    )
    parser.add_argument(
        "--input_file",
        type=Path,
        required=True,
        help="输入 BVH 文件路径。",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=None,
        help=(
            "输出 BVH 文件路径。未指定时，自动保存为："
            "原文件名_clip_起始帧_结束帧.bvh"
        ),
    )
    parser.add_argument(
        "--start_frame",
        type=int,
        required=True,
        help="起始帧编号，从 0 开始，并且包含该帧。",
    )
    parser.add_argument(
        "--end_frame",
        type=int,
        required=True,
        help="结束帧编号，并且包含该帧；设置为 -1 表示最后一帧。",
    )
    return parser.parse_args()


def get_line_ending(line: str) -> str:
    """取得当前文本行原有的换行符。"""
    if line.endswith("\r\n"):
        return "\r\n"
    if line.endswith("\n"):
        return "\n"
    if line.endswith("\r"):
        return "\r"
    return ""


def read_bvh_lines(input_path: Path) -> list[str]:
    """读取 BVH，并保留原文件的换行格式。"""
    try:
        with input_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            return file.readlines()
    except UnicodeDecodeError as error:
        raise ValueError(
            f"无法按 UTF-8 编码读取 BVH 文件：{input_path}"
        ) from error


def locate_motion_section(
    lines: list[str],
) -> tuple[int, int, int, int]:
    """
    定位 MOTION、Frames 和 Frame Time，并统计骨架通道数量。

    返回：
        motion_line_index
        frames_line_index
        frame_time_line_index
        channel_count
    """
    motion_line_index = -1
    frames_line_index = -1
    frame_time_line_index = -1
    channel_count = 0

    channel_pattern = re.compile(
        r"^\s*CHANNELS\s+(\d+)\b",
        flags=re.IGNORECASE,
    )
    frames_pattern = re.compile(
        r"^\s*Frames\s*:\s*(\d+)\s*$",
        flags=re.IGNORECASE,
    )
    frame_time_pattern = re.compile(
        r"^\s*Frame\s+Time\s*:\s*"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$",
        flags=re.IGNORECASE,
    )

    for line_index, line in enumerate(lines):
        stripped = line.strip()

        # MOTION 之前的 CHANNELS 数量之和，就是每帧应有的数据数量。
        if motion_line_index < 0:
            channel_match = channel_pattern.match(line)
            if channel_match:
                channel_count += int(channel_match.group(1))

        if stripped.upper() == "MOTION":
            motion_line_index = line_index
            continue

        if motion_line_index >= 0 and frames_line_index < 0:
            if frames_pattern.match(stripped):
                frames_line_index = line_index
                continue

        if frames_line_index >= 0 and frame_time_line_index < 0:
            if frame_time_pattern.match(stripped):
                frame_time_line_index = line_index
                break

    if motion_line_index < 0:
        raise ValueError("BVH 文件中没有找到 MOTION 段。")
    if frames_line_index < 0:
        raise ValueError("BVH 文件中没有找到 Frames 行。")
    if frame_time_line_index < 0:
        raise ValueError("BVH 文件中没有找到 Frame Time 行。")
    if channel_count <= 0:
        raise ValueError("BVH 骨架中没有读取到有效的 CHANNELS。")

    return (
        motion_line_index,
        frames_line_index,
        frame_time_line_index,
        channel_count,
    )


def parse_declared_frame_count(frames_line: str) -> int:
    """读取 Frames 行中声明的总帧数。"""
    match = re.match(
        r"^\s*Frames\s*:\s*(\d+)\s*$",
        frames_line.strip(),
        flags=re.IGNORECASE,
    )
    if match is None:
        raise ValueError(f"无法解析 Frames 行：{frames_line.rstrip()}")
    return int(match.group(1))


def collect_motion_frames(
    lines: list[str],
    frame_time_line_index: int,
    channel_count: int,
) -> list[str]:
    """
    收集动作帧并检查每帧数据数量。

    标准 BVH 文件要求一行对应一帧。
    """
    frame_lines: list[str] = []

    for source_line_number, line in enumerate(
        lines[frame_time_line_index + 1 :],
        start=frame_time_line_index + 2,
    ):
        # 跳过动作区域中的空行，避免将空行误认为动作帧。
        if not line.strip():
            continue

        values = line.split()
        if len(values) != channel_count:
            raise ValueError(
                "动作帧数据数量与骨架通道数量不一致：\n"
                f"  文件行号：{source_line_number}\n"
                f"  应有数值：{channel_count}\n"
                f"  实际数值：{len(values)}"
            )

        # 保留动作帧原有的文本精度和数值格式。
        frame_lines.append(line)

    return frame_lines


def build_output_path(
    input_path: Path,
    output_path: Path | None,
    start_frame: int,
    end_frame: int,
) -> Path:
    """生成输出文件路径。"""
    if output_path is not None:
        return output_path.expanduser().resolve()

    output_name = (
        f"{input_path.stem}_clip_{start_frame}_{end_frame}"
        f"{input_path.suffix}"
    )
    return input_path.with_name(output_name).resolve()


def replace_frame_count_line(
    original_line: str,
    new_frame_count: int,
) -> str:
    """仅替换 Frames 行中的帧数，保留缩进和换行格式。"""
    indentation = original_line[
        : len(original_line) - len(original_line.lstrip())
    ]
    line_ending = get_line_ending(original_line)
    return f"{indentation}Frames: {new_frame_count}{line_ending}"


def clip_bvh(
    input_path: Path,
    output_path: Path | None,
    start_frame: int,
    end_frame: int,
) -> Path:
    """执行 BVH 连续帧裁剪。"""
    input_path = input_path.expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在：{input_path}")
    if not input_path.is_file():
        raise ValueError(f"输入路径不是文件：{input_path}")
    if input_path.suffix.lower() != ".bvh":
        raise ValueError(f"输入文件不是 .bvh 文件：{input_path}")
    if start_frame < 0:
        raise ValueError("start_frame 不能小于 0。")
    if end_frame < -1:
        raise ValueError("end_frame 只能是 -1 或大于等于 0 的帧编号。")

    lines = read_bvh_lines(input_path)
    (
        _motion_line_index,
        frames_line_index,
        frame_time_line_index,
        channel_count,
    ) = locate_motion_section(lines)

    declared_frame_count = parse_declared_frame_count(
        lines[frames_line_index]
    )
    motion_frames = collect_motion_frames(
        lines,
        frame_time_line_index,
        channel_count,
    )
    actual_frame_count = len(motion_frames)

    if actual_frame_count != declared_frame_count:
        raise ValueError(
            "BVH 声明帧数与实际动作帧数不一致：\n"
            f"  Frames 声明：{declared_frame_count}\n"
            f"  实际读取：{actual_frame_count}"
        )

    # -1 表示使用最后一帧。
    effective_end_frame = (
        actual_frame_count - 1
        if end_frame == -1
        else end_frame
    )

    if start_frame >= actual_frame_count:
        raise ValueError(
            f"start_frame 超出范围。有效范围为 "
            f"0 到 {actual_frame_count - 1}。"
        )
    if effective_end_frame >= actual_frame_count:
        raise ValueError(
            f"end_frame 超出范围。有效范围为 "
            f"0 到 {actual_frame_count - 1}，或者使用 -1。"
        )
    if start_frame > effective_end_frame:
        raise ValueError(
            "start_frame 不能大于 end_frame。"
        )

    # Python 切片右端不包含，因此结束位置需要加 1。
    clipped_frames = motion_frames[
        start_frame : effective_end_frame + 1
    ]
    clipped_frame_count = len(clipped_frames)

    final_output_path = build_output_path(
        input_path=input_path,
        output_path=output_path,
        start_frame=start_frame,
        end_frame=effective_end_frame,
    )

    if final_output_path == input_path:
        raise ValueError(
            "输出文件不能与输入文件相同，避免覆盖原始 BVH。"
        )

    final_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 完整保留 HIERARCHY、MOTION 和 Frame Time，
    # 只更新 Frames 数量并替换动作帧区间。
    output_lines = lines[: frame_time_line_index + 1]
    output_lines[frames_line_index] = replace_frame_count_line(
        original_line=output_lines[frames_line_index],
        new_frame_count=clipped_frame_count,
    )
    output_lines.extend(clipped_frames)

    with final_output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        file.writelines(output_lines)

    print("===== BVH 裁剪完成 =====")
    print(f"输入文件：{input_path}")
    print(f"原始总帧数：{actual_frame_count}")
    print(f"保留帧范围：{start_frame} 到 {effective_end_frame}")
    print(f"裁剪后帧数：{clipped_frame_count}")
    print(f"每帧通道数：{channel_count}")
    print(f"输出文件：{final_output_path}")

    return final_output_path


def main() -> None:
    """程序入口。"""
    args = parse_args()
    clip_bvh(
        input_path=args.input_file,
        output_path=args.output_file,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )


if __name__ == "__main__":
    main()