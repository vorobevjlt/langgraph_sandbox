import base64
import pprint
from pathlib import Path
from moviepy import VideoFileClip
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from settings import settings

CHUNK_DURATION = 60

def video_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

llm = ChatGoogleGenerativeAI(
    model="gemini-3-pro-preview",
    api_key=settings.OPENAI_API_KEY,
    temperature=0.1,
    max_retries=2,
    base_url="https://api.proxyapi.ru/google"
)

message_video_file_id = HumanMessage(content=[
    {"type": "text", "text": "Whats going on in this video"},
    {"type": "media", "file_id": "file-abc123"},
])

if __name__ == "__main__":
    current_file = Path(__file__).resolve()
    video_path = current_file.parent.parent / "resourses" / "my_video.mp4"

    clip = VideoFileClip(str(video_path))
    duration = int(clip.duration)

    results = []

    for start in range(0, duration, CHUNK_DURATION):
        end = min(start + CHUNK_DURATION, duration)

        chunk = clip.with_start(start).with_end(end)

        tmp_path = video_path.parent / f"chunk_{start}_{end}.mp4"
        chunk.write_videofile(
            str(tmp_path),
            codec="libx264",
            audio_codec="aac",
            logger=None
        )

        video_data = video_to_base64(tmp_path)

        recognize_message = HumanMessage(content=[
            {
                "type": "text",
                "text": f"Describe what happens in this video segment ({start}-{end} seconds)"
            },
            {
                "type": "video",
                "base64": video_data,
                "mime_type": "video/mp4",
            }
        ])

        result = llm.invoke([recognize_message])
        results.append({
            "segment": f"{start}-{end}",
            "description": result.content
        })

        tmp_path.unlink()  # cleanup temp file

    pprint.pprint(results)