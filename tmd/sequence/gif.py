class GifExporter(BaseExporter):
    """
    Exporter for creating animated GIFs from height map sequences.
    """
    
    def export(self, **kwargs) -> Optional[str]:
        """
        Expects the following kwargs:
            - frames: List of 2D numpy arrays (required)
            - output_file: Destination path for the GIF (defaults to 'output.gif')
            - fps: Frames per second (default: 10.0)
            - colormap: Matplotlib colormap name (default: 'terrain')
            - loop: Loop count for the GIF (default: 0 for infinite)
            - optimize: Whether to optimize the GIF (default: True)
            - duration: Duration per frame in milliseconds (optional)
            - show_progress: Whether to display progress (default: True)
            - Additional kwargs passed to PIL.Image.save
        """
        # Import dependencies and helper functions
        from tmd.utils.lib_utils import import_optional_dependency, check_dependencies
        from tmd.utils.files import ensure_directory_exists, get_progress_bar
        
        # Check dependencies
        dependencies = ['matplotlib.pyplot', 'matplotlib.cm', 'PIL.Image']
        dependency_status = check_dependencies(dependencies)
        HAS_MATPLOTLIB = dependency_status['matplotlib.pyplot'] and dependency_status['matplotlib.cm']
        HAS_PIL = dependency_status['PIL.Image']
        
        if not HAS_MATPLOTLIB or not HAS_PIL:
            logger.error("Required packages (matplotlib and Pillow) not available")
            return None
        
        # Import required modules
        plt = import_optional_dependency('matplotlib.pyplot')
        cm = import_optional_dependency('matplotlib.cm')
        Image = import_optional_dependency('PIL.Image')
        
        # Retrieve parameters
        frames: List[np.ndarray] = kwargs.get('frames', [])
        output_file: str = kwargs.get('output_file', 'output.gif')
        fps: float = kwargs.get('fps', 10.0)
        colormap: str = kwargs.get('colormap', 'terrain')
        loop: int = kwargs.get('loop', 0)
        optimize: bool = kwargs.get('optimize', True)
        duration: Optional[float] = kwargs.get('duration', None)
        show_progress: bool = kwargs.get('show_progress', True)
        extra_kwargs: Dict[str, Any] = kwargs.get('extra_kwargs', {})

        if not frames:
            logger.error("No frames provided for GIF export")
            return None
        
        try:
            # Ensure output directory exists
            ensure_directory_exists(os.path.dirname(os.path.abspath(output_file)))
            if not output_file.lower().endswith('.gif'):
                output_file += '.gif'
            
            # Calculate duration from fps if not provided
            if duration is None:
                duration = int(1000 / fps)  # in milliseconds
            
            # Normalize data across all frames for consistent color mapping
            all_min = min(np.nanmin(frame) for frame in frames)
            all_max = max(np.nanmax(frame) for frame in frames)
            norm_range = all_max - all_min if all_max > all_min else 1.0
            
            cmap = cm.get_cmap(colormap)
            gif_frames = []
            
            frame_iterator = get_progress_bar(frames, desc="Creating GIF") if show_progress else frames
            
            for frame in frame_iterator:
                norm_frame = (frame - all_min) / norm_range
                rgba_img = (cmap(norm_frame) * 255).astype(np.uint8)
                gif_frames.append(Image.fromarray(rgba_img))
            
            if gif_frames:
                gif_frames[0].save(
                    output_file,
                    format='GIF',
                    append_images=gif_frames[1:],
                    save_all=True,
                    duration=duration,
                    loop=loop,
                    optimize=optimize,
                    **extra_kwargs
                )
                logger.info(f"GIF animation with {len(frames)} frames saved to {output_file}")
                return output_file
            else:
                logger.error("No frames were processed for GIF export")
                return None
        except Exception as e:
            logger.error(f"Error exporting to GIF: {e}")
            return None