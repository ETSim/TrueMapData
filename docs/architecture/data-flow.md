# TMD Data Flow

This document outlines the data flow through the TMD library, showing how data moves between different components during typical operations.

## Core Data Flow Sequence

The standard data flow through the TMD library follows this sequence:

```mermaid
flowchart TD
    TMDFile[TMD File] --> Processor
    Processor --> HeightMap
    Processor --> Metadata

    HeightMap --> Processing[Processing Operations]
    HeightMap --> Filtering[Filtering Operations]
    HeightMap --> Analysis[Analysis Operations]

    Processing --> ProcessedMap[Processed Height Map]
    Filtering --> FilteredMap[Filtered Height Map]
    Analysis --> AnalysisResults[Analysis Results]

    ProcessedMap --> Visualization
    FilteredMap --> Visualization
    AnalysisResults --> Visualization

    ProcessedMap --> Export
    FilteredMap --> Export
    AnalysisResults --> Export

    Export --> ImageFormats[Image Formats]
    Export --> ModelFormats[3D Model Formats]
    Export --> DataFormats[Data Formats]

    Visualization --> StaticViz[Static Visualizations]
    Visualization --> InteractiveViz[Interactive Visualizations]
```

## Key Data Objects

### TMD File

The starting point - a binary file containing:

- Height map data
- Metadata about physical dimensions
- File format information

### Processed Data

After parsing the TMD file, the data is represented as:

1. **Height Map**: A 2D NumPy array of floating-point values representing surface heights
2. **Metadata**: A dictionary containing information like:
   - Physical dimensions (width, height)
   - Units (µm, nm, etc.)
   - Comments from the original file
   - File format version

## Processing Pipeline

A typical processing pipeline looks like this:

1. **Load TMD File**: The processor loads and parses the binary TMD file
2. **Extract Height Map**: The height map is extracted and converted to a NumPy array
3. **Process/Filter**: Various operations can be applied to the height map
   - Gaussian filtering for smoothing
   - Thresholding for outlier removal
   - Cropping for region-of-interest analysis
4. **Analysis**: Calculate metrics like roughness or extract cross-sections
5. **Visualization**: Create visual representations of the data
6. **Export**: Save the results in various formats

## Data Transformations

Throughout the pipeline, the height map undergoes various transformations:

1. **Initial Processing**:
   - Raw binary data → NumPy array
   - Metadata extraction

2. **Height Map Operations**:
   - Filtering (Gaussian, median, etc.)
   - Geometric operations (crop, rotate)
   - Statistical operations (normalize, threshold)

3. **Export Transformations**:
   - Height map → Displacement map (grayscale image)
    P3[Section Location] -.->|Configure| D
    end

```

## Error Handling Flow

This diagram shows how errors are handled during processing:

```mermaid
flowchart TD
    A[Process Start] -->|Read File| B{File Valid?}
    B -->|Yes| C[Parse Header]
    B -->|No| Z[Error: File Not Found]

    C -->|Parse Complete| D{Header Valid?}
    D -->|Yes| E[Parse Data]
    D -->|No| Y[Error: Invalid Header]

    E -->|Parse Complete| F{Data Valid?}
    F -->|Yes| G[Processing Complete]
    F -->|No| X[Error: Invalid Data]

    Z --> Error
    Y --> Error
    X --> Error

    subgraph "Error Handling"
    Error -->|Log Error| H[Error Log]
    Error -->|Return None| I[Null Result]
    end

    classDef success fill:#dfd,stroke:#333,stroke-width:1px;
    classDef error fill:#fdd,stroke:#333,stroke-width:1px;
    classDef process fill:#ddf,stroke:#333,stroke-width:1px;
    classDef decision fill:#ffd,stroke:#333,stroke-width:1px;

    class A,C,E,G process;
    class B,D,F decision;
    class Z,Y,X,Error error;
    class G success;
```

## Data Type Flow

This diagram shows how data types flow through the system:

```mermaid
flowchart LR
    A[Binary File] -->|Read| B[Raw Bytes]
    B -->|Parse Header| C[Metadata Dict]
    B -->|Parse Data| D[1D Float Array]
    D -->|Reshape| E[2D Height Map]
    E -->|Analyze| F[Processed Data]

    classDef fileType fill:#fcf,stroke:#333,stroke-width:1px;
    classDef rawType fill:#cff,stroke:#333,stroke-width:1px;
    classDef structType fill:#ffc,stroke:#333,stroke-width:1px;
    classDef arrayType fill:#cfc,stroke:#333,stroke-width:1px;

    class A fileType;
    class B rawType;
    class C structType;
    class D,E,F arrayType;
```

## State Diagram for TMDProcessor

This diagram shows the state transitions of a TMDProcessor object:

```mermaid
stateDiagram-v2
    [*] --> Initialized: Create Processor
    Initialized --> Processed: process()
    Processed --> WithHeightMap: get_height_map()
    Processed --> WithMetadata: get_metadata()
    Processed --> WithStats: get_stats()
    WithHeightMap --> Exported: export
    WithHeightMap --> Visualized: visualize
    WithHeightMap --> Analyzed: analyze
    WithStats --> ReportGenerated: generate_report
    Processed --> ErrorState: error occurs
    ErrorState --> Initialized: reset
    Initialized --> [*]: dispose
```

These diagrams provide a comprehensive view of how data flows through the TMD library, helping users understand its architecture and processing pipeline.
