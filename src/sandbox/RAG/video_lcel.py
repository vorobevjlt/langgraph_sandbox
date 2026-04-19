import base64
from pathlib import Path
from moviepy import VideoFileClip
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from settings import settings

current_file = Path(__file__).resolve()
video_path = current_file.parent.parent.parent / "resourses" / "my_video.mp4"

llm = ChatGoogleGenerativeAI(
    model="gemini-3-pro-preview",
    api_key=settings.OPENAI_API_KEY,
    temperature=0.1,
    max_retries=2,
    base_url="https://api.proxyapi.ru/google"
)
def video_chunks_base64(
    video_path: Path,
    chunk_seconds: int = 60
):
    clip = VideoFileClip(str(video_path))
    duration = int(clip.duration)

    for start in range(0, duration, chunk_seconds):
        end = min(start + chunk_seconds, duration)

        chunk = clip.with_start(start).with_end(end)

        tmp_path = video_path.parent / f"chunk_{start}_{end}.mp4"
        chunk.write_videofile(
            str(tmp_path),
            codec="libx264",
            audio_codec="aac",
            logger=None
        )

        with open(tmp_path, "rb") as f:
            yield {
                "start": start,
                "end": end,
                "video_b64": base64.b64encode(f.read()).decode("utf-8")
            }

        tmp_path.unlink()

def chunk_to_message(chunk):
    return [HumanMessage(content=[
        {
            "type": "text",
            "text": f"Describe what happens in this video segment ({chunk['start']}–{chunk['end']} seconds)"
        },
        {
            "type": "video",
            "base64": chunk["video_b64"],
            "mime_type": "video/mp4",
        }
    ])]

describe_chunk_chain = (
    RunnableLambda(chunk_to_message)
    | llm
    | RunnableLambda(lambda r: r.content)
)

chunks = list(video_chunks_base64(video_path))

chunk_descriptions = describe_chunk_chain.batch(
    chunks,
    config={"max_concurrency": 4}
)

glue_prompt = ChatPromptTemplate.from_messages([
    ("system", "You summarize videos from sequential segment descriptions."),
    ("human",
     "Below are descriptions of consecutive video segments.\n\n"
     "{segments}\n\n"
     "Create a coherent, chronological description of the full video.")
])

glue_chain = (
    glue_prompt
    | llm
    | RunnableLambda(lambda r: r.content)
)

final_description = glue_chain.invoke({
    "segments": "\n".join(
        f"Segment {i+1}: {desc}"
        for i, desc in enumerate(chunk_descriptions)
    )
})

print(final_description)