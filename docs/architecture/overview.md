# TMD Architecture Overview

This page provides a high-level overview of the TMD library architecture and its major components.

## Overall Architecture

The TMD library is organized into several layers, from low-level file processing to high-level visualization and analysis tools.

```mermaid
graph TD
    A[TMD Files] --> B[File Processing Layer]
    B --> C[Core Processing Layer]
    C --> D1[Analysis Tools]
    C --> D2[Visualization Tools]
    C --> D3[Export Tools]
    D1 --> E[Results/Insights]
    D2 --> E
    D3 --> F[Output Files]
    
    classDef core fill:#f9f,stroke:#333,stroke-width:2px;
    classDef input fill:#bbf,stroke:#333,stroke-width:1px;
    classDef output fill:#bfb,stroke:#333,stroke-width:1px;
    class B,C core;
    class A input;
    class E,F output;
```

## Key Components

The library consists of several key components that work together to process and analyze TMD files:

```mermaid
graph LR
    TMDProcessor[TMDProcessor] --> Utils[Utils Module]
    TMDProcessor --> Filter[Filter Module]
    TMDProcessor --> Plotters[Plotters Module]
    TMDProcessor --> Exporters[Exporters Module]
    
    Utils --> ProcessingUtils[Processing Utils]
    Utils --> FileUtils[File Utils]
    
    Filter --> WavinessFilter[Waviness/Roughness]
    Filter --> SlopeCalculation[Gradient/Slope]
    
    Plotters --> MatplotlibPlotter[Matplotlib]
    Plotters --> OtherPlotters[Other Plotters]
    
    Exporters --> STLExporter[STL Export]
    Exporters --> NPYExporter[NumPy Export]
    Exporters --> ImageExporter[Image Export]
    
    classDef main fill:#f96,stroke:#333,stroke-width:2px;
    classDef module fill:#9cf,stroke:#333,stroke-width:1px;
    classDef submodule fill:#fcf,stroke:#333,stroke-width:1px;
    class TMDProcessor main;
    class Utils,Filter,Plotters,Exporters module;
    class ProcessingUtils,FileUtils,WavinessFilter,SlopeCalculation,MatplotlibPlotter,OtherPlotters,STLExporter,NPYExporter,ImageExporter submodule;
```

## Processing Pipeline

The TMD file processing pipeline consists of several stages from input to output:

```mermaid
flowchart TD
    A[Input TMD File] --> B[File Reading]
    B --> C[Metadata Extraction]
    B --> D[Height Map Extraction]
    C --> E[Data Validation]
    D --> E
    E --> F{Processing Required?}
    F -->|Yes| G[Apply Filters/Processing]
    F -->|No| H[Analysis]
    G --> H
    H --> I[Visualization/Export]
    
    classDef process fill:#d1c7ff,stroke:#333,stroke-width:1px;
    classDef decision fill:#ffcccc,stroke:#333,stroke-width:1px;
    classDef output fill:#ccffcc,stroke:#333,stroke-width:1px;
    
    class A,B,C,D,E,G,H process;
    class F decision;
    class I output;
```

## Class Relationships

The following diagram shows the key classes and their relationships:

```mermaid
classDiagram
    class TMDProcessor {
        +file_path: str
        +data: dict
        +process()
        +get_height_map()
        +get_metadata()
        +get_stats()
    }
    
    class HeightMap {
        +array: ndarray
        +metadata: dict
        +apply_filter()
        +calculate_stats()
    }
    
    class FilterUtils {
        +extract_waviness()
        +extract_roughness()
        +calculate_gradient()
    }
    
    class ExportTools {
        +export_to_stl()
        +export_to_npy()
        +export_to_image()
    }
    
    class VisualizationTools {
        +plot_3d()
        +plot_heatmap()
        +plot_profile()
    }
    
    TMDProcessor --> HeightMap: creates
    HeightMap --> FilterUtils: uses
    HeightMap --> ExportTools: uses
    HeightMap --> VisualizationTools: uses
```

## File Format Structure

The TMD file format consists of a header section and a data section:

```mermaid
graph TD
    subgraph "TMD File Structure"
    A[File Header] --> A1[Version Identifier]
    A --> A2[Comment Section]
    A --> A3[Dimensions]
    A --> A4[Spatial Parameters]
    
    B[Data Section] --> B1[Height Map Data]
    B1 --> B2[Row-major Float32 Array]
    end
    
    classDef header fill:#ffddaa,stroke:#333,stroke-width:1px;
    classDef data fill:#aaddff,stroke:#333,stroke-width:1px;
    
    class A,A1,A2,A3,A4 header;
    class B,B1,B2 data;
```

This overview provides a foundation for understanding how the TMD library is structured and how its components interact with each other.
