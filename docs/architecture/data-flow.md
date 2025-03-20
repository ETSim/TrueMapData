# Data Flow in the TMD Library

This document illustrates how data flows through the TMD library during different operations.

## Basic Processing Flow

The following diagram shows the basic flow of data when processing a TMD file:

```mermaid
flowchart TD
    A[TMD File] -->|Read| B[Binary Data]
    B -->|Parse Header| C[Metadata]
    B -->|Parse Data Section| D[Raw Height Data]
    D -->|Reshape| E[Height Map Array]
    C --> F[Processor Data Dictionary]
    E --> F
    F -->|Get Height Map| G[Height Map for Analysis]
    F -->|Get Stats| H[Statistical Summary]
    G -->|Apply Filters| I[Processed Height Map]
    I -->|Export| J[Output Files]
    I -->|Visualize| K[Plots/Graphs]
```

## Waviness and Roughness Separation

This diagram illustrates how a height map is separated into waviness and roughness components:

```mermaid
flowchart LR
    A[Height Map] -->|Gaussian Filter| B[Waviness Component]
    A -->|Subtraction| C{Difference}
    B --> C
    C -->|Result| D[Roughness Component]

    subgraph "Parameters"
    P[Sigma Value] -.->|Controls Filter| B
    end

    B -->|RMS Calculation| E[RMS Waviness]
    D -->|RMS Calculation| F[RMS Roughness]
```

## Export Process Flow

The following diagram shows the data flow during the export process:

```mermaid
flowchart TD
    A[Height Map] -->|STL Export| B[3D Mesh Generation]
    B -->|Add Base| C[Complete 3D Model]
    C -->|Write File| D[STL File]

    A -->|Image Export| E[Color Mapping]
    E -->|Add Colorbar| F[Visualization]
    F -->|Write File| G[Image File]

    A -->|NumPy Export| H[Serialize Data]
    H -->|Write File| I[NPY/NPZ File]

    subgraph "Export Parameters"
    P1[Scale Factors] -.->|Adjust| B
    P2[Color Maps] -.->|Configure| E
    P3[Compression] -.->|Configure| H
    end
```

## Processing Pipeline for Surface Analysis

This diagram shows the data flow for surface analysis operations:

```mermaid
flowchart TD
    A[Height Map] -->|Gradient Calculation| B[X & Y Gradients]
    B -->|Magnitude Calculation| C[Slope Map]

    A -->|Extract Cross-Section| D[Height Profile]
    D -->|Plot| E[Profile Graph]

    A -->|Threshold| F[Thresholded Map]
    F -->|Region Selection| G[ROI Analysis]

    A -->|Apply Filter| H[Filtered Map]
    H -->|Statistical Analysis| I[RMS/Roughness Values]

    subgraph "Analysis Parameters"
    P1[Filter Parameters] -.->|Configure| H
    P2[Threshold Values] -.->|Configure| F
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
