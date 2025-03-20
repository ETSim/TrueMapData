# TMD Architecture: Component Diagram

This document provides a visual representation of the TMD library's architecture, showing the key components and their relationships.

## Component Overview

```mermaid
graph TD
    User[User/Application] --> Processor
    
    subgraph Core
        Processor[TMDProcessor]
        Utils[FileUtils]
    end
    
    subgraph Processing
        Filter[FilterModule]
        Processing[ProcessingModule]
    end
    
    subgraph Visualization
        MatPlotLib[MatplotlibPlotter]
        Plotly[PlotlyPlotter]
    end
    
    subgraph Export
        ImageExporter[ImageExporter]
        ModelExporter[3DModelExporter]
        CompressionExporter[CompressionExporter]
    end
    
    Processor --> Utils
    Processor --> Processing
    Processor --> Filter
    
    Processing --> Filter
    
    Processor --> MatPlotLib
    Processor --> Plotly
    
    Processor --> ImageExporter
    Processor --> ModelExporter
    Processor --> CompressionExporter
    
    classDef core fill:#f96,stroke:#333,stroke-width:2px;
    classDef processing fill:#9cf,stroke:#333,stroke-width:2px;
    classDef visualization fill:#f9f,stroke:#333,stroke-width:2px;
    classDef export fill:#9f9,stroke:#333,stroke-width:2px;
    
    class Processor,Utils core;
    class Filter,Processing processing;
    class MatPlotLib,Plotly visualization;
    class ImageExporter,ModelExporter,CompressionExporter export;
```

## Data Flow Diagram

```mermaid
flowchart TD
    subgraph Input
        TMDFile[TMD File]
    end
    
    subgraph Processing
        Processor[TMD Processor]
        HeightMap[Height Map Extraction]
        Filtering[Filtering & Analysis]
        Manipulation[Manipulation Operations]
    end
    
    subgraph Output
        VisualizationOutput[Visualization]
        ExportOutput[Export Formats]
    end
    
    TMDFile --> Processor
    Processor --> HeightMap
    HeightMap --> Filtering
    HeightMap --> Manipulation
    
    Filtering --> VisualizationOutput
    Manipulation --> VisualizationOutput
    
    HeightMap --> ExportOutput
    Filtering --> ExportOutput
    Manipulation --> ExportOutput
    
    classDef input fill:#ffd, stroke:#333, stroke-width:2px;
    classDef process fill:#dff, stroke:#333, stroke-width:2px;
    classDef output fill:#dfd, stroke:#333, stroke-width:2px;
    
    class TMDFile input;
    class Processor,HeightMap,Filtering,Manipulation process;
    class VisualizationOutput,ExportOutput output;
```

## Processing Sequence

This sequence diagram shows the complete process flow when using the TMD library:

```mermaid
sequenceDiagram
    actor User
    participant Processor as TMDProcessor
    participant Utils as FileUtils
    participant Filter as FilterModule
    participant Manipulation as Processing
    participant Export as Exporters
    participant Viz as Visualization
    
    User->>Processor: 1. Create processor(file_path)
    User->>Processor: 2. process()
    
    activate Processor
    Processor->>Utils: 3. process_tmd_file()
    activate Utils
    Utils-->>Processor: 4. return metadata, height_map
    deactivate Utils
    Processor-->>User: 5. return processed data
    deactivate Processor
    
    User->>Manipulation: 6. manipulate height map
    activate Manipulation
    Note over Manipulation: crop, rotate, threshold, etc.
    Manipulation-->>User: 7. return modified height map
    deactivate Manipulation
    
    User->>Filter: 8. apply filters to height map
    activate Filter
    Note over Filter: gaussian, waviness, roughness, etc.
    Filter-->>User: 9. return filtered height map
    deactivate Filter
    
    User->>Viz: 10. visualize height map
    activate Viz
    Note over Viz: 2D/3D plots, cross-sections, etc.
    Viz-->>User: 11. return visualization
    deactivate Viz
    
    User->>Export: 12. export to various formats
    activate Export
    Note over Export: images, 3D models, NumPy arrays
    Export-->>User: 13. export files
    deactivate Export
```

## Component Descriptions

### Core Components
- **TMDProcessor**: Main class for loading and processing TMD files
- **FileUtils**: Handles file I/O operations and TMD format parsing

### Processing Components
- **FilterModule**: Implements various filters and analysis algorithms
- **ProcessingModule**: Provides tools for manipulating height maps

### Visualization Components
- **MatplotlibPlotter**: Creates static visualizations using Matplotlib
- **PlotlyPlotter**: Creates interactive visualizations using Plotly

### Export Components
- **ImageExporter**: Exports height maps to various image formats
- **ModelExporter**: Exports height maps to 3D model formats (STL, OBJ, PLY)
- **CompressionExporter**: Exports height maps to NumPy formats
