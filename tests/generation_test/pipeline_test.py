import pytest
import asyncio
from aichat.pipeline.tracks import AudioOutTrack, VideoOutTrack

# Integration test of speech generation

@pytest.mark.asyncio
async def test_pipeline_connect(mock_gpu_components):
    """pipeline should set input and output queue when called connect"""
    from aichat.pipeline.generation import GenerationPipeline
    
    pipeline = GenerationPipeline(config={})
    
    a_track = AudioOutTrack()
    v_track = VideoOutTrack()

    assert pipeline.a_queue is None, "audio queue should be none before connect is called."
    assert pipeline.v_queue is None, "video queue should be none before connect is called."
    
    pipeline.connect(a_track, v_track)
    
    assert pipeline.a_queue is a_track.queue, "audio queue should be set to track's queue."
    assert pipeline.v_queue is v_track.queue, "video queue should be set to track's queue."
    

@pytest.mark.asyncio
async def test_pipeline_generate(mock_gpu_components):
    """pipeline should call tts component with only with one complete sentences."""
    pass
    # TODO: This test will be work when work on improvement phase. Idea is split the sentences
    # until we have single complete sentence, we call warmup to warmup tts component.
    # as soon as we have complete single sentence, we start working tts for speech while llm
    # continue to write next sentence, This is attempt to speed up the generation process.
    
@pytest.mark.asyncio
async def test_pipeline_call_tts_warmup(mock_gpu_components):
    """pipeline should call tts component's warmup before complete sentence is received."""
    
    from aichat.pipeline.generation import GenerationPipeline
    
    pipeline = GenerationPipeline(config={})
    
    a_track = AudioOutTrack()
    v_track = VideoOutTrack()
    pipeline.connect(a_track, v_track)
    
    await pipeline.generate("hello there")
    
    assert mock_gpu_components["Orpheus"].warmup.call_count == 3
    
@pytest.mark.asyncio
async def test_pipeline_call_tts_with_full(mock_gpu_components):
    """pipeline should called tts component's synthesis after full generation is done."""
    
    from aichat.pipeline.generation import GenerationPipeline
    
    pipeline = GenerationPipeline(config={})
    
    a_track = AudioOutTrack()
    v_track = VideoOutTrack()
    pipeline.connect(a_track, v_track)
    
    await pipeline.generate("hello there")
    
    mock_gpu_components["Orpheus"].synthesize.assert_called_once_with("text1 text2 text3")
    
    
@pytest.mark.asyncio
async def test_pipeline_should_convert_pcm_to_frames(mock_gpu_components):
    """pipeline should convert pcm bytes form tts to frames"""
    
    from aichat.pipeline.generation import GenerationPipeline
    
    pipeline = GenerationPipeline(config={})
    
    a_track = AudioOutTrack()
    v_track = VideoOutTrack()
    pipeline.connect(a_track, v_track)
    
    await pipeline.generate("hello there")
    
    frame1 = await a_track.recv()
    frame2 = await a_track.recv()
    
    
    assert bytes(frame1.planes[0]) != b'\x00' * 320, "frame1 should contain non-zero audio data" # type: ignore
    assert bytes(frame2.planes[0]) != b'\x00' * 320, "frame2 should contain non-zero audio data" # type: ignore
    
    
@pytest.mark.asyncio
async def test_pipeline_generation_should_be_cancellable(mock_gpu_components):
    """pipeline generation should be able to cancel"""
    pass
    