from moviepy.editor import VideoFileClip
import os
import pympi

videoname="video2"                      # Which videos?

def cut_video_in_intervals(video_path, start_sec, end_sec, interval_sec, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    video = VideoFileClip(video_path)

    current = start_sec
    count = 0

    while current + interval_sec <= end_sec:
        subclip = video.subclip(current, current + interval_sec)
        filename = f"{videoname}_{count:03d}.mp4"  # 숫자만 출력 (예: 000.mp4)
        output_path = os.path.join(output_dir, filename)
        subclip.write_videofile(output_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)
        current += interval_sec
        count += 1

    video.close()

def trim_eaf(eaf_path, start_sec, end_sec, interval_sec, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    original_eaf = pympi.Elan.Eaf(eaf_path)

    current = start_sec
    idx = 0

    while current + interval_sec <= end_sec:
        interval_start = current
        interval_end = current + interval_sec

        start_ms = int(interval_start * 1000)
        end_ms = int(interval_end * 1000)

        # 출력 파일 경로
        eaf_out_path = os.path.join(output_dir, f"{idx:03d}.eaf")

        # EAF 자르기 + 시간 shift
        trimmed_eaf = pympi.Elan.Eaf()
        for tier in original_eaf.get_tier_names():
            trimmed_eaf.add_tier(tier)
            for annotation in original_eaf.get_annotation_data_for_tier(tier):
                start, end, value = annotation[:3]
                if start >= start_ms and end <= end_ms:
                    shifted_start = start - start_ms
                    shifted_end = end - start_ms
                    trimmed_eaf.add_annotation(tier, shifted_start, shifted_end, value)

        trimmed_eaf.to_file(eaf_out_path)

        current += interval_sec
        idx += 1

output=f"{videoname}/"   
start_sec=1176                          # starttime
end_sec=1506                            # endtime
interval_sec=10                         # don't change

cut_video_in_intervals(
    video_path=f"{videoname}/angle_0_final_output.mp4",
    start_sec=start_sec,      
    end_sec=end_sec,        
    interval_sec=interval_sec,
    output_dir=output
)

"""trim_eaf(
    eaf_path=f"{videoname}/eaf/annotation.eaf",
    start_sec=start_sec,
    end_sec=end_sec,
    interval_sec=interval_sec,
    output_dir=f"{videoname}/annotations"
)"""