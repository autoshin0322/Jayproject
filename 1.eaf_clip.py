import os
import pympi

def split_eaf_segments(eaf_path, start_sec, end_sec, interval_sec, output_root):
    original_eaf = pympi.Elan.Eaf(eaf_path)

    current = start_sec
    idx = 0

    def iter_alignable_triples(eaf, tier_name):
        for ann in eaf.get_annotation_data_for_tier(tier_name):
            if len(ann) >= 3:
                start_ms, end_ms, value = ann[0], ann[1], ann[2]
                if isinstance(start_ms, (int, float)) and isinstance(end_ms, (int, float)):
                    yield int(start_ms), int(end_ms), value

    while current + interval_sec <= end_sec:
        slice_start = current
        slice_end = current + interval_sec
        start_ms = int(slice_start * 1000)
        end_ms = int(slice_end * 1000)

        # ✅ 바로 annotations 폴더에 저장
        out_name = f"{idx:03d}.eaf"
        eaf_out_path = os.path.join(output_root, out_name)

        trimmed = pympi.Elan.Eaf()
        for tier in original_eaf.get_tier_names():
            if tier not in trimmed.get_tier_names():
                trimmed.add_tier(tier)

            for ann_start, ann_end, value in iter_alignable_triples(original_eaf, tier):
                if ann_end > start_ms and ann_start < end_ms:
                    clip_start = max(ann_start, start_ms) - start_ms
                    clip_end = min(ann_end, end_ms) - start_ms
                    if clip_end > clip_start:
                        trimmed.add_annotation(tier, clip_start, clip_end, value)

        trimmed.to_file(eaf_out_path)
        print(f"✅ 저장 완료: {eaf_out_path}")

        current += interval_sec
        idx += 1


# 실행 예시
video="video3"

split_eaf_segments(
    eaf_path=f"{video}/eaf/annotation.eaf",
    start_sec=660,   # 9분 20초
    end_sec=920,    # 25분 20초
    interval_sec=10, # 10초 단위
    output_root=f"{video}/annotations"
)