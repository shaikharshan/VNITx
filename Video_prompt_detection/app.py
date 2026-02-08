import httpx
import streamlit as st


st.set_page_config(page_title="Visual Security Engine", layout="wide")
st.title("Visual Security Engine Demo")

tab_image, tab_video = st.tabs(["Image", "Video"])

with st.sidebar:
    st.header("API Settings")
    mode = st.selectbox("API mode", ["gateway", "split"], index=0)
    gateway_url = st.text_input("Gateway URL", value="http://localhost:8000")
    engine_d_url = st.text_input("Engine D URL", value="http://localhost:8001")
    engine_e_url = st.text_input("Engine E URL", value="http://localhost:8002")
    video_url = st.text_input("Video API URL", value="http://localhost:8010")
    st.caption("Gateway mode calls a single API. Split mode calls D/E separately.")
    st.header("Performance")
    run_ocr = st.checkbox("Show OCR output", value=True)
    run_injection = st.checkbox("Run prompt-injection model", value=True)
    run_cross_modal = st.checkbox("Run cross-modal check", value=True)
    run_caption = st.checkbox("Run BLIP caption alignment", value=True)
    if run_injection and not run_ocr:
        st.info("OCR is required for prompt-injection. Enabling OCR display.")
        run_ocr = True

with tab_image:
    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
    transcript = st.text_area("Audio transcript (optional)", value="a cat sitting on a ledge")
    run_clicked = st.button("Run image analysis", type="primary")

    if run_clicked and not uploaded:
        st.warning("Please upload an image to continue.")

    if run_clicked and uploaded:
        image_bytes = uploaded.read()
        st.image(image_bytes, caption="Uploaded image", use_container_width=True)

        with st.spinner("Calling APIs for analysis..."):
            text_payload = {}
            injection_result = {"skipped": True}
            cross_modal_result = {"skipped": True}

            if mode == "gateway":
                try:
                    response = httpx.post(
                        f"{gateway_url.rstrip('/')}/analyze",
                        files={"image": (uploaded.name, image_bytes, uploaded.type or "image/jpeg")},
                        data={
                            "audio_transcript": transcript,
                            "run_caption": str(run_caption).lower(),
                            "deep": str(run_injection).lower(),
                        },
                        timeout=300,
                    )
                    response.raise_for_status()
                except Exception as exc:
                    st.error("Gateway API call failed. Is it running on the configured URL?")
                    st.exception(exc)
                    st.stop()
                payload = response.json()
                text_payload = payload.get("ocr", {})
                injection_result = payload.get("injection", {})
                cross_modal_result = payload.get("cross_modal", {})
                ocr_vs_image = payload.get("ocr_vs_image", {})
                caption_alignment = payload.get("caption_alignment", {})
                final_score = payload.get("final_score")
            else:
                if run_injection or run_ocr:
                    try:
                        response_d = httpx.post(
                            f"{engine_d_url.rstrip('/')}/analyze_d",
                            files={"image": (uploaded.name, image_bytes, uploaded.type or "image/jpeg")},
                            data={"deep": str(run_injection).lower()},
                            timeout=300,
                        )
                        response_d.raise_for_status()
                    except Exception as exc:
                        st.error("Engine D API call failed. Is it running on the configured URL?")
                        st.exception(exc)
                    else:
                        payload_d = response_d.json()
                        text_payload = payload_d.get("ocr", {})
                        injection_result = payload_d.get("injection", {})

                if run_cross_modal:
                    try:
                        response_e = httpx.post(
                            f"{engine_e_url.rstrip('/')}/analyze_e",
                            files={"image": (uploaded.name, image_bytes, uploaded.type or "image/jpeg")},
                            data={
                                "audio_transcript": transcript,
                                "ocr_text": text_payload.get("normalized_text", ""),
                                "run_caption": str(run_caption).lower(),
                            },
                            timeout=300,
                        )
                        response_e.raise_for_status()
                    except Exception as exc:
                        st.error("Engine E API call failed. Is it running on the configured URL?")
                        st.exception(exc)
                    else:
                        payload_e = response_e.json()
                        cross_modal_result = payload_e.get("cross_modal", {})
                        ocr_vs_image = payload_e.get("ocr_vs_image", {})
                        caption_alignment = payload_e.get("caption_alignment", {})
                else:
                    ocr_vs_image = {"skipped": True}
                    caption_alignment = {"skipped": True}
                final_score = None

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("OCR Output")
                if not run_ocr:
                    st.info("OCR display disabled.")
                else:
                    st.text_area("Raw text", value=text_payload.get("raw_text", ""), height=150)
                    st.text_area(
                        "Normalized text", value=text_payload.get("normalized_text", ""), height=120
                    )

            with col2:
                st.subheader("Engine D: Prompt Injection")
                st.json(injection_result)
                st.subheader("Engine E: Cross-Modal Consistency")
                st.json(cross_modal_result)
                st.subheader("OCR vs Image (CLIP)")
                st.json(ocr_vs_image)
                st.subheader("Caption Alignment (BLIP)")
                st.json(caption_alignment)
                if final_score is not None:
                    st.subheader("Final Risk Score")
                    st.metric("final_score", final_score)

with tab_video:
    video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    video_transcript = st.text_area("Video transcript (optional)", value="")
    target_fps = st.slider("Target FPS", min_value=1, max_value=10, value=5)
    max_frames = st.number_input("Max frames (0 = no limit)", min_value=0, value=0)
    run_vision_deepfake = st.checkbox("Run vision deepfake detector", value=True)
    run_avsync = st.checkbox("Run audio-visual sync check", value=True)
    log_frames = st.checkbox("Write per-frame log file", value=True)
    run_video = st.button("Run video analysis", type="primary")
    st.divider()
    st.subheader("Webcam")
    camera_index = st.number_input("Camera index", min_value=0, value=0)
    duration_sec = st.slider("Duration (seconds)", min_value=5, max_value=30, value=10)
    run_webcam = st.button("Run webcam analysis", type="secondary")

    if run_video and not video_file:
        st.warning("Please upload a video to continue.")

    if run_video and video_file:
        video_bytes = video_file.read()
        with st.spinner("Calling Video API for analysis..."):
            try:
                response = httpx.post(
                    f"{video_url.rstrip('/')}/analyze_video",
                    files={"video": (video_file.name, video_bytes, video_file.type or "video/mp4")},
                    data={
                        "audio_transcript": video_transcript,
                        "target_fps": float(target_fps),
                        "max_frames": int(max_frames) or None,
                        "run_injection": str(run_injection).lower(),
                        "run_cross_modal": str(run_cross_modal).lower(),
                        "run_caption": str(run_caption).lower(),
                        "run_vision_deepfake": str(run_vision_deepfake).lower(),
                        "run_avsync": str(run_avsync).lower(),
                        "log_frames": str(log_frames).lower(),
                    },
                    timeout=600,
                )
                response.raise_for_status()
            except Exception as exc:
                st.error("Video API call failed. Is it running on the configured URL?")
                st.exception(exc)
            else:
                payload = response.json()
                st.subheader("Summary")
                st.json(payload.get("summary", {}))
                summary = payload.get("summary", {})
                if summary.get("verdict"):
                    st.subheader("Video Verdict")
                    st.metric("verdict", summary.get("verdict"))
                    st.caption(f"Criteria: {summary.get('verdict_criteria', {})}")
                if payload.get("action"):
                    st.subheader("Mitigation Action")
                    st.metric("action", payload.get("action"))
                    st.caption(f"Explanations: {payload.get('explanations', [])}")
                st.subheader("Top Risky Frames")
                st.dataframe(payload.get("top_risky_frames_flat", []))
                st.subheader("Timeline (Flattened)")
                st.dataframe(payload.get("timeline_flat", []))
                if payload.get("log_path"):
                    st.info(f"Frame log written to: {payload['log_path']}")

    if run_webcam:
        with st.spinner("Calling Webcam API for analysis..."):
            try:
                response = httpx.post(
                    f"{video_url.rstrip('/')}/analyze_webcam",
                    data={
                        "camera_index": int(camera_index),
                        "duration_sec": float(duration_sec),
                        "target_fps": float(target_fps),
                        "run_injection": str(run_injection).lower(),
                        "run_cross_modal": str(run_cross_modal).lower(),
                        "run_caption": str(run_caption).lower(),
                        "run_vision_deepfake": str(run_vision_deepfake).lower(),
                        "run_avsync": str(run_avsync).lower(),
                        "log_frames": str(log_frames).lower(),
                    },
                    timeout=600,
                )
                response.raise_for_status()
            except Exception as exc:
                st.error("Webcam API call failed. Is it running on the configured URL?")
                st.exception(exc)
            else:
                payload = response.json()
                st.subheader("Webcam Summary")
                st.json(payload.get("summary", {}))
                summary = payload.get("summary", {})
                if summary.get("verdict"):
                    st.subheader("Webcam Verdict")
                    st.metric("verdict", summary.get("verdict"))
                    st.caption(f"Criteria: {summary.get('verdict_criteria', {})}")
                if payload.get("action"):
                    st.subheader("Mitigation Action")
                    st.metric("action", payload.get("action"))
                    st.caption(f"Explanations: {payload.get('explanations', [])}")
                st.subheader("Webcam Top Risky Frames")
                st.dataframe(payload.get("top_risky_frames_flat", []))
                st.subheader("Webcam Timeline (Flattened)")
                st.dataframe(payload.get("timeline_flat", []))
                if payload.get("log_path"):
                    st.info(f"Frame log written to: {payload['log_path']}")
