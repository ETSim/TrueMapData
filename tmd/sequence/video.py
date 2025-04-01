class VideoExporter(BaseExporter):
    """
    Exporter for creating videos (e.g., MP4) from height map sequences.
    """
    
    def export(self, **kwargs) -> Optional[str]:
        """
        Expects the following kwargs:
            - frames: List of 2D numpy arrays (required)
            - output_file: Destination path for the video file (defaults to 'output.mp4')
            - fps: Frames per second (default: 30.0)
            - colormap: Matplotlib colormap name (default: 'terrain')
            - dpi: Resolution for frames (default: 100)
            - quality: Optional video quality (0-10, higher is better)
            - show_progress: Whether to display a progress bar (default: True)
            - bitrate: Optional bitrate for video encoding
            - codec: Optional video codec (e.g., 'libx264')
            - Additional kwargs passed to the animation save function
        """
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation
            from matplotlib import cm
            from tqdm import tqdm
            
            frames: List[np.ndarray] = kwargs.get('frames', [])
            output_file: str = kwargs.get('output_file', 'output.mp4')
            fps: float = kwargs.get('fps', 30.0)
            colormap: str = kwargs.get('colormap', 'terrain')
            dpi: int = kwargs.get('dpi', 100)
            quality: Optional[int] = kwargs.get('quality', None)
            show_progress: bool = kwargs.get('show_progress', True)
            bitrate: Optional[int] = kwargs.get('bitrate', None)
            codec: Optional[str] = kwargs.get('codec', None)
            extra_kwargs: Dict[str, Any] = kwargs.get('extra_kwargs', {})
            
            if not frames:
                logger.error("No frames provided for video export")
                return None
            
            # Use non-interactive backend
            matplotlib.use('Agg')
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            if not output_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                output_file += '.mp4'
            
            fig, ax = plt.subplots(figsize=(10, 8))
            all_min = min(np.min(frame) for frame in frames)
            all_max = max(np.max(frame) for frame in frames)
            norm_range = all_max - all_min if all_max > all_min else 1.0
            
            norm_frame = (frames[0] - all_min) / norm_range
            im = ax.imshow(norm_frame, cmap=colormap, animated=True)
            ax.axis('off')
            plt.colorbar(im, ax=ax)
            title = ax.text(0.5, 1.05, 'Frame: 0', ha="center", transform=ax.transAxes)
            
            def update_frame(i):
                norm_frame = (frames[i] - all_min) / norm_range
                im.set_array(norm_frame)
                title.set_text(f'Frame: {i}')
                return [im, title]
            
            if show_progress:
                class TqdmAnimation(animation.FuncAnimation):
                    def __init__(self, *args, **kwargs):
                        self.n_frames = len(frames)
                        self.pbar = tqdm(total=self.n_frames, desc="Creating video")
                        super().__init__(*args, **kwargs)
                    
                    def _step(self, *args, **kwargs):
                        result = super()._step(*args, **kwargs)
                        self.pbar.update(1)
                        return result
                    
                    def finish(self):
                        self.pbar.close()
                
                anim = TqdmAnimation(
                    fig, update_frame, frames=len(frames),
                    interval=1000/fps, blit=True
                )
            else:
                anim = animation.FuncAnimation(
                    fig, update_frame, frames=len(frames),
                    interval=1000/fps, blit=True
                )
            
            writer_kwargs = {}
            if bitrate:
                writer_kwargs['bitrate'] = bitrate
            if quality:
                writer_kwargs['quality'] = quality / 10.0
            
            writer = animation.FFMpegWriter(
                fps=fps, codec=codec,
                metadata=dict(title="Height Map Animation"),
                **writer_kwargs
            )
            
            save_kwargs = {'writer': writer, 'dpi': dpi}
            save_kwargs.update(extra_kwargs)
            
            anim.save(output_file, **save_kwargs)
            if show_progress and hasattr(anim, 'finish'):
                anim.finish()
            plt.close(fig)
            
            logger.info(f"Video saved to {output_file}")
            return output_file
        
        except Exception as e:
            logger.error(f"Error exporting to video: {e}")
            return None