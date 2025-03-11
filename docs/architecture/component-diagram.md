# TMD Component Diagram

This page shows the relationships between various components of the TMD library.

## Core Components

The TMD library consists of several core components that work together to provide a complete workflow for processing and analyzing TMD files.

```mermaid
graph TB
    subgraph Core
    TMDProcessor[TMDProcessor]
    end
    
    subgraph Utils
    FileUtils[File Utilities]
    ProcessingUtils[Processing Utilities]
    MetadataUtils[Metadata Utilities]
    end
    
    subgraph Analysis
    Filter[Filtering Module]
    Stats[Statistics Module]
    Gradient[Gradient Analysis]
    end
    
    subgraph Output
    ExportSTL[STL Exporter]
    ExportNPY[NumPy Exporter]
    ExportImage[Image Exporter]
    PlotterMatplotlib[Matplotlib Plotter]
    end
    
    TMDProcessor --> FileUtils
    TMDProcessor --> ProcessingUtils
    TMDProcessor --> MetadataUtils
    TMDProcessor --> Filter
    TMDProcessor --> Stats
    TMDProcessor --> Gradient
    TMDProcessor --> ExportSTL
    TMDProcessor --> ExportNPY
    TMDProcessor --> ExportImage
    TMDProcessor --> PlotterMatplotlib
    
    FileUtils --> ProcessingUtils
    Stats --> Filter
    Filter --> Gradient

    classDef core fill:#f96,stroke:#333,stroke-width:2px;
    classDef util fill:#9cf,stroke:#333,stroke-width:1px;
    classDef analysis fill:#fcf,stroke:#333,stroke-width:1px;
    classDef output fill:#cf9,stroke:#333,stroke-width:1px;
    
    class TMDProcessor core;
    class FileUtils,ProcessingUtils,MetadataUtils util;
    class Filter,Stats,Gradient analysis;
    class ExportSTL,ExportNPY,ExportImage,PlotterMatplotlib output;
```

## Component Dependencies

This diagram shows the dependencies between different components:

```mermaid
flowchart TD
    subgraph Core
    A[TMDProcessor]
    end
    
    subgraph External Dependencies
    B1[NumPy]
    B2[SciPy]
    B3[Matplotlib]
    end
    
    subgraph TMD Modules
    C1[tmd.utils]
    C2[tmd.filters]
    C3[tmd.exporters]
    C4[tmd.plotters]
    end
    
    A --> C1
    A --> C2
    A --> C3
    A --> C4
    
    C1 --> B1
    C1 --> B2
    C2 --> B1
    C2 --> B2
    C3 --> B1
    C4 --> B1
    C4 --> B3
    
    classDef core fill:#f96,stroke:#333,stroke-width:2px;
    classDef external fill:#9cf,stroke:#333,stroke-width:1px;
    classDef module fill:#fcf,stroke:#333,stroke-width:1px;
    
    class A core;
    class B1,B2,B3 external;
    class C1,C2,C3,C4 module;
```

## Physical Component Structure

This diagram shows the physical file structure of the library:

```mermaid
graph TD
    A[tmd Package] --> B1[__init__.py]
    A --> B2[processor.py]
    A --> B3[_version.py]
    
    A --> C[utils Module]
    C --> C1[__init__.py]
    C --> C2[utils.py]
    C --> C3[filter.py]
    C --> C4[processing.py]
    C --> C5[metadata.py]
    
    A --> D[exporters Module]
    D --> D1[__init__.py]
    D --> D2[stl.py]
    D --> D3[numpy.py]
    D --> D4[image.py]
    
    A --> E[plotters Module]
    E --> E1[__init__.py]
    E --> E2[matplotlib.py]
    E --> E3[plotly.py]
    E --> E4[seaborn.py]
    
    classDef folder fill:#f9d,stroke:#333,stroke-width:2px;
    classDef file fill:#ddf,stroke:#333,stroke-width:1px;
    
    class A,C,D,E folder;
    class B1,B2,B3,C1,C2,C3,C4,C5,D1,D2,D3,D4,E1,E2,E3,E4 file;
```

## Component Interfaces

This diagram shows the main interfaces between components:

```mermaid
classDiagram
    class TMDProcessor {
        +file_path: str
        +data: Dict
        +debug: bool
        +process()
        +get_height_map()
        +get_metadata()
        +get_stats()
        +export_metadata()
    }
    
    class FileReader {
        <<Interface>>
        +read_file(path: str)
        +parse_header()
        +parse_data()
    }
    
    class Exporter {
        <<Interface>>
        +export(data, path: str)
    }
    
    class STLExporter {
        +convert_heightmap_to_stl()
    }
    
    class NPYExporter {
        +export_to_npy()
        +export_to_npz()
    }
    
    class ImageExporter {
        +export_to_png()
        +export_to_jpg()
    }
    
    class Plotter {
        <<Interface>>
        +plot(data)
        +save(path: str)
    }
    
    class MatplotlibPlotter {
        +plot_height_map_matplotlib()
        +plot_2d_heatmap_matplotlib()
    }
    
    FileReader <|-- TMDProcessor
    Exporter <|.. STLExporter
    Exporter <|.. NPYExporter
    Exporter <|.. ImageExporter
    Plotter <|.. MatplotlibPlotter
    TMDProcessor --> Exporter
    TMDProcessor --> Plotter
```

## Processing Sequence

This sequence diagram shows the process of loading and analyzing a TMD file:

```mermaid
sequenceDiagram
    actor User
    participant Processor as TMDProcessor
    participant Utils as FileUtils
    participant Filter as FilterModule
    participant Export as Exporter
    
    User->>Processor: Create processor(file_path)
    User->>Processor: process()
    activate Processor
    Processor->>Utils: process_tmd_file(file_path)
    activate Utils
    Utils-->>Processor: metadata, height_map
    deactivate Utils
    Processor->>Processor: store data
    deactivate Processor
    
    User->>Processor: get_height_map()
    Processor-->>User: height_map
    
    User->>Filter: calculate_rms_roughness(height_map)
    activate Filter
    Filter->>Filter: extract_roughness(height_map)
    Filter-->>User: roughness_value
    deactivate Filter
    
    User->>Export: convert_heightmap_to_stl(height_map)
    activate Export
    Export-->>User: stl_file_path
    deactivate Export
```

These diagrams provide a comprehensive view of the TMD library's architecture and component relationships, helping developers understand how the different parts work together.
