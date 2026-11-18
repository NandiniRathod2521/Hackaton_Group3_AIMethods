import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";
import { FPS, type Frame } from "../pose/coco";

// Plays the local video (object URL; it never leaves the browser) with the analysed skeleton
// drawn on top, so the viewer can see which pose was used at every moment.

const EDGES = [[0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12],
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]];
const MIN_DRAW_CONF = 0.2;

export interface VideoPlayerHandle { playFrom: (seconds: number) => void }

interface Props {
  src: string;
  frames: Frame[];
  width: number;
  height: number;
  showSkeleton: boolean;
}

type RVFCVideo = HTMLVideoElement & {
  requestVideoFrameCallback?: (cb: (now: number, meta: { mediaTime: number }) => void) => number;
  cancelVideoFrameCallback?: (id: number) => void;
};

export const VideoPlayer = forwardRef<VideoPlayerHandle, Props>(function VideoPlayer(
  { src, frames, width, height, showSkeleton }, ref,
) {
  const videoRef = useRef<RVFCVideo>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useImperativeHandle(ref, () => ({
    playFrom(seconds: number) {
      const v = videoRef.current;
      if (!v) return;
      v.currentTime = seconds;
      v.play().catch(() => undefined);
      v.scrollIntoView({ behavior: "smooth", block: "center" });
    },
  }), []);

  useEffect(() => {
    const v = videoRef.current, c = canvasRef.current;
    if (!v || !c) return;
    const ctx = c.getContext("2d")!;
    const draw = (t: number) => {
      ctx.clearRect(0, 0, c.width, c.height);
      if (!showSkeleton || !frames.length) return;
      const f = frames[Math.min(frames.length - 1, Math.max(0, Math.round(t * FPS)))];
      ctx.lineWidth = Math.max(2, width / 200);
      ctx.strokeStyle = "#00c8ff";
      ctx.fillStyle = "#ffffff";
      for (const [a, b] of EDGES) {
        if (f[a][2] < MIN_DRAW_CONF || f[b][2] < MIN_DRAW_CONF) continue;
        ctx.beginPath(); ctx.moveTo(f[a][0], f[a][1]); ctx.lineTo(f[b][0], f[b][1]); ctx.stroke();
      }
      for (const [px, py, p] of f) {
        if (p < MIN_DRAW_CONF) continue;
        ctx.beginPath(); ctx.arc(px, py, Math.max(3, width / 150), 0, Math.PI * 2); ctx.fill();
      }
    };
    let id = 0;
    const loop = (_n: number, meta: { mediaTime: number }) => {
      draw(meta.mediaTime);
      id = v.requestVideoFrameCallback!(loop);
    };
    const onSeek = () => draw(v.currentTime);
    if (v.requestVideoFrameCallback) id = v.requestVideoFrameCallback(loop);
    v.addEventListener("seeked", onSeek);
    v.addEventListener("timeupdate", onSeek);
    draw(v.currentTime);
    return () => {
      if (id && v.cancelVideoFrameCallback) v.cancelVideoFrameCallback(id);
      v.removeEventListener("seeked", onSeek);
      v.removeEventListener("timeupdate", onSeek);
    };
  }, [frames, showSkeleton, width]);

  return (
    <div className="player" style={{ aspectRatio: `${width} / ${height}`, width: `min(100%, calc(75vh * ${width} / ${height}))` }}>
      <video ref={videoRef} src={src} controls muted playsInline preload="auto" />
      <canvas ref={canvasRef} width={width} height={height} aria-hidden="true" />
    </div>
  );
});
