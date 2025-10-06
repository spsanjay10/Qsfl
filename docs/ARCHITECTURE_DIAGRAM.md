# QSFL-CAAD System Architecture

This document provides comprehensive architectural diagrams for the Quantum-Safe Federated Learning with Client Anomaly and Attack Detection (QSFL-CAAD) system.

## Table of Contents
1. [High-Level System Architecture](#high-level-system-architecture)
2. [Component Architecture](#component-architecture)
3. [Data Flow Architecture](#data-flow-architecture)
4. [Security Architecture](#security-architecture)
5. [Deployment Architecture](#deployment-architecture)

---

## High-Level System Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        C1[Honest Client 1]
        C2[Honest Client 2]
        C3[Honest Client N]
        M1[Malicious Client]
    end
    
    subgraph "API Gateway Layer"
        API[API Gateway<br/>Rate Limiting & Auth]
    end
    
    subgraph "Core System Layer"
        subgraph "Security Module"
            PQC[Post-Quantum<br/>Cryptography]
            AUTH[Authentication<br/>Service]
            REV[Revocation<br/>Manager]
        end
        
        subgraph "Detection Module"
            AD[Anomaly<br/>Detector]
            FE[Feature<br/>Extractor]
            SHAP[SHAP<br/>Explainer]
            REP[Reputation<br/>Manager]
        end
        
        subgraph "Federated Learning Module"
            FLS[FL Server]
            AGG[Model<br/>Aggregator]
            MUH[Update<br/>Handler]
        end
        
        subgraph "Monitoring Module"
            LOG[Security<br/>Logger]
            MET[Metrics<br/>Collector]
            ALERT[Alert<br/>Manager]
        end
    end
    
    subgraph "Data Layer"
        DB[(Database)]
        CACHE[(Redis Cache)]
        STORAGE[(Model Storage)]
    end
    
    subgraph "Monitoring Stack"
        PROM[Prometheus]
        GRAF[Grafana]
        ELK[ELK Stack]
    end
    
    C1 & C2 & C3 & M1 --> API
    API --> AUTH
    AUTH --> PQC
    AUTH --> REV
    API --> FLS
    FLS --> MUH
    MUH --> AD
    AD --> FE
    AD --> SHAP
    AD --> REP
    MUH --> AGG
    FLS --> LOG
    FLS --> MET
    MET --> ALERT
    
    AUTH --> DB
    REV --> DB
    REP --> DB
    FLS --> STORAGE
    REP --> CACHE
    
    LOG --> ELK
    MET --> PROM
    PROM --> GRAF
    
    style PQC fill:#ff9999
    style AUTH fill:#ff9999
    style AD fill:#99ccff
    style FLS fill:#99ff99
    style LOG fill:#ffcc99
```

---

## Component Architecture

### Core Components Interaction

```mermaid
graph LR
    subgraph "Client Request Flow"
        CR[Client Request]
    end
    
    subgraph "Authentication Pipeline"
        A1[Validate Credentials]
        A2[Check Revocation]
        A3[Verify Signature]
        A4[Issue Token]
    end
    
    subgraph "Model Update Pipeline"
        M1[Receive Update]
        M2[Extract Features]
        M3[Detect Anomaly]
        M4[Update Reputation]
        M5[Aggregate Model]
    end
    
    subgraph "Response Flow"
        R1[Log Event]
        R2[Collect Metrics]
        R3[Send Response]
    end
    
    CR --> A1
    A1 --> A2
    A2 --> A3
    A3 --> A4
    A4 --> M1
    M1 --> M2
    M2 --> M3
    M3 --> M4
    M4 --> M5
    M5 --> R1
    R1 --> R2
    R2 --> R3
    
    style A1 fill:#ffcccc
    style M3 fill:#ccddff
    style M5 fill:#ccffcc
```

### Module Dependencies

```mermaid
graph TD
    subgraph "Application Layer"
        API[API Endpoints]
        UI[Web Dashboard]
        CLI[CLI Interface]
    end
    
    subgraph "Service Layer"
        FLS[Federated Learning Server]
        AUTH[Authentication Service]
        AD[Anomaly Detection Service]
    end
    
    subgraph "Core Layer"
        PQC[PQ Crypto Manager]
        AGG[Model Aggregator]
        FE[Feature Extractor]
        REP[Reputation Manager]
    end
    
    subgraph "Infrastructure Layer"
        DB[Database Layer]
        CACHE[Cache Layer]
        LOG[Logging Layer]
        MET[Metrics Layer]
    end
    
    API --> FLS
    API --> AUTH
    UI --> API
    CLI --> API
    
    FLS --> AUTH
    FLS --> AD
    FLS --> AGG
    
    AUTH --> PQC
    AD --> FE
    AD --> REP
    
    FLS --> LOG
    FLS --> MET
    AUTH --> LOG
    AD --> LOG
    
    REP --> CACHE
    AUTH --> DB
    FLS --> DB
    
    style FLS fill:#90EE90
    style AUTH fill:#FFB6C1
    style AD fill:#87CEEB
```

---

## Data Flow Architecture

### Training Round Data Flow

```mermaid
sequenceDiagram
    participant C as Client
    participant API as API Gateway
    participant AUTH as Auth Service
    participant FLS as FL Server
    participant AD as Anomaly Detector
    participant REP as Reputation Manager
    participant AGG as Aggregator
    participant DB as Database
    
    C->>API: Submit Model Update
    API->>AUTH: Authenticate Client
    AUTH->>AUTH: Verify Signature
    AUTH-->>API: Authentication Result
    
    alt Authentication Failed
        API-->>C: 401 Unauthorized
    else Authentication Success
        API->>FLS: Forward Update
        FLS->>AD: Analyze Update
        AD->>AD: Extract Features
        AD->>AD: Compute Anomaly Score
        AD-->>FLS: Anomaly Score
        
        FLS->>REP: Update Reputation
        REP->>DB: Store Reputation
        
        alt High Anomaly Score
            FLS->>FLS: Reduce Weight/Reject
            FLS-->>C: Update Rejected/Weighted
        else Normal Score
            FLS->>AGG: Add to Aggregation Pool
            AGG->>AGG: Aggregate Models
            AGG->>DB: Store Global Model
            AGG-->>FLS: Aggregation Complete
            FLS-->>C: Update Accepted
        end
    end
```

### Security Event Flow

```mermaid
sequenceDiagram
    participant SYS as System Component
    participant LOG as Security Logger
    participant MET as Metrics Collector
    participant ALERT as Alert Manager
    participant PROM as Prometheus
    participant ADMIN as Administrator
    
    SYS->>LOG: Log Security Event
    LOG->>LOG: Format & Enrich Event
    LOG->>DB: Persist Event
    
    SYS->>MET: Record Metric
    MET->>PROM: Export Metric
    
    alt Critical Event
        LOG->>ALERT: Trigger Alert
        ALERT->>ALERT: Evaluate Severity
        ALERT->>ADMIN: Send Notification (Email/SMS)
        ALERT->>DB: Store Alert
    end
    
    PROM->>PROM: Evaluate Alert Rules
    alt Threshold Exceeded
        PROM->>ADMIN: Send Alert
    end
```

---

## Security Architecture

### Post-Quantum Cryptography Layer

```mermaid
graph TB
    subgraph "Client Side"
        CKP[Client Keypair<br/>Dilithium]
        CSIGN[Sign Update]
    end
    
    subgraph "Transport Layer"
        TLS[TLS 1.3<br/>+ PQ KEX]
        ENC[Encrypted Channel]
    end
    
    subgraph "Server Side"
        SKP[Server Keypair<br/>Kyber + Dilithium]
        VERIFY[Verify Signature]
        DECRYPT[Decrypt Data]
    end
    
    subgraph "Key Management"
        KS[Key Storage<br/>HSM/Vault]
        KR[Key Rotation<br/>Policy]
        REV[Revocation List]
    end
    
    CKP --> CSIGN
    CSIGN --> TLS
    TLS --> ENC
    ENC --> VERIFY
    VERIFY --> SKP
    ENC --> DECRYPT
    
    SKP --> KS
    CKP --> KS
    KS --> KR
    VERIFY --> REV
    
    style CKP fill:#ff9999
    style SKP fill:#ff9999
    style TLS fill:#ffcc99
    style KS fill:#99ccff
```

### Authentication & Authorization Flow

```mermaid
stateDiagram-v2
    [*] --> Unregistered
    Unregistered --> Registered: Register Client
    Registered --> Authenticated: Authenticate
    Authenticated --> Active: Token Issued
    Active --> Authenticated: Token Refresh
    Active --> Quarantined: High Anomaly Score
    Quarantined --> Active: Manual Review
    Active --> Revoked: Revoke Credentials
    Revoked --> [*]
    
    note right of Registered
        Credentials Issued
        Public/Private Keys
    end note
    
    note right of Quarantined
        Temporary Suspension
        Reduced Influence
    end note
    
    note right of Revoked
        Permanent Ban
        Credentials Invalid
    end note
```

### Anomaly Detection Pipeline

```mermaid
graph LR
    subgraph "Input Stage"
        MU[Model Update]
    end
    
    subgraph "Feature Extraction"
        FE1[Statistical Features]
        FE2[Gradient Features]
        FE3[Distribution Features]
        FE4[Temporal Features]
    end
    
    subgraph "Detection Stage"
        IF[Isolation Forest]
        SCORE[Anomaly Score]
    end
    
    subgraph "Explanation Stage"
        SHAP[SHAP Values]
        EXP[Feature Importance]
    end
    
    subgraph "Action Stage"
        REP[Update Reputation]
        DEC[Decision Engine]
        ACT[Action: Allow/Reduce/Reject]
    end
    
    MU --> FE1 & FE2 & FE3 & FE4
    FE1 & FE2 & FE3 & FE4 --> IF
    IF --> SCORE
    SCORE --> SHAP
    SHAP --> EXP
    SCORE --> REP
    SCORE --> DEC
    REP --> DEC
    DEC --> ACT
    
    style IF fill:#87CEEB
    style SHAP fill:#98FB98
    style DEC fill:#FFB6C1
```

---

## Deployment Architecture

### Docker Compose Deployment

```mermaid
graph TB
    subgraph "Load Balancer"
        NGINX[Nginx<br/>Reverse Proxy]
    end
    
    subgraph "Application Tier"
        APP1[QSFL-CAAD<br/>Instance 1]
        APP2[QSFL-CAAD<br/>Instance 2]
        APP3[QSFL-CAAD<br/>Instance N]
    end
    
    subgraph "Worker Tier"
        CELERY1[Celery Worker 1]
        CELERY2[Celery Worker 2]
        BEAT[Celery Beat<br/>Scheduler]
    end
    
    subgraph "Data Tier"
        POSTGRES[(PostgreSQL)]
        REDIS[(Redis)]
        MINIO[(MinIO<br/>Object Storage)]
    end
    
    subgraph "Monitoring Tier"
        PROM[Prometheus]
        GRAF[Grafana]
        ES[Elasticsearch]
        KIB[Kibana]
    end
    
    NGINX --> APP1 & APP2 & APP3
    APP1 & APP2 & APP3 --> POSTGRES
    APP1 & APP2 & APP3 --> REDIS
    APP1 & APP2 & APP3 --> MINIO
    
    CELERY1 & CELERY2 --> REDIS
    CELERY1 & CELERY2 --> POSTGRES
    BEAT --> REDIS
    
    APP1 & APP2 & APP3 --> PROM
    APP1 & APP2 & APP3 --> ES
    PROM --> GRAF
    ES --> KIB
    
    style NGINX fill:#90EE90
    style POSTGRES fill:#87CEEB
    style REDIS fill:#FFB6C1
    style PROM fill:#FFD700
```

### Kubernetes Deployment

```mermaid
graph TB
    subgraph "Ingress Layer"
        ING[Ingress Controller<br/>NGINX/Traefik]
    end
    
    subgraph "Application Namespace"
        subgraph "API Deployment"
            API1[API Pod 1]
            API2[API Pod 2]
            API3[API Pod 3]
        end
        
        subgraph "Worker Deployment"
            W1[Worker Pod 1]
            W2[Worker Pod 2]
        end
        
        SVC[Service<br/>Load Balancer]
    end
    
    subgraph "Data Namespace"
        PG[PostgreSQL<br/>StatefulSet]
        RD[Redis<br/>StatefulSet]
        PVC1[PVC: DB Data]
        PVC2[PVC: Cache]
    end
    
    subgraph "Monitoring Namespace"
        PROM[Prometheus]
        GRAF[Grafana]
        ALERT[AlertManager]
    end
    
    subgraph "Storage"
        SC[StorageClass]
        PV1[PV: Database]
        PV2[PV: Models]
    end
    
    ING --> SVC
    SVC --> API1 & API2 & API3
    API1 & API2 & API3 --> PG
    API1 & API2 & API3 --> RD
    W1 & W2 --> PG
    W1 & W2 --> RD
    
    PG --> PVC1
    RD --> PVC2
    PVC1 --> PV1
    PVC2 --> SC
    
    API1 & API2 & API3 --> PROM
    W1 & W2 --> PROM
    PROM --> GRAF
    PROM --> ALERT
    
    style ING fill:#90EE90
    style SVC fill:#87CEEB
    style PG fill:#FFB6C1
    style PROM fill:#FFD700
```

### Cloud Architecture (AWS Example)

```mermaid
graph TB
    subgraph "Edge Layer"
        CF[CloudFront CDN]
        WAF[AWS WAF]
    end
    
    subgraph "VPC - Public Subnet"
        ALB[Application<br/>Load Balancer]
        NAT[NAT Gateway]
    end
    
    subgraph "VPC - Private Subnet"
        subgraph "Application Tier"
            ECS1[ECS Task 1]
            ECS2[ECS Task 2]
            ECS3[ECS Task N]
        end
        
        subgraph "Data Tier"
            RDS[(RDS PostgreSQL<br/>Multi-AZ)]
            ELASTIC[(ElastiCache<br/>Redis)]
            S3[(S3 Bucket<br/>Model Storage)]
        end
    end
    
    subgraph "Monitoring & Security"
        CW[CloudWatch]
        XR[X-Ray]
        SM[Secrets Manager]
        KMS[KMS<br/>Encryption]
    end
    
    CF --> WAF
    WAF --> ALB
    ALB --> ECS1 & ECS2 & ECS3
    
    ECS1 & ECS2 & ECS3 --> RDS
    ECS1 & ECS2 & ECS3 --> ELASTIC
    ECS1 & ECS2 & ECS3 --> S3
    ECS1 & ECS2 & ECS3 --> NAT
    
    ECS1 & ECS2 & ECS3 --> CW
    ECS1 & ECS2 & ECS3 --> XR
    ECS1 & ECS2 & ECS3 --> SM
    
    RDS --> KMS
    S3 --> KMS
    
    style CF fill:#FF9900
    style ALB fill:#FF9900
    style RDS fill:#527FFF
    style S3 fill:#569A31
```

---

## Network Architecture

### Network Topology

```mermaid
graph TB
    subgraph "Internet"
        CLIENT[Clients]
    end
    
    subgraph "DMZ Zone"
        FW1[Firewall]
        LB[Load Balancer]
    end
    
    subgraph "Application Zone - VLAN 10"
        APP[Application Servers<br/>10.0.10.0/24]
    end
    
    subgraph "Data Zone - VLAN 20"
        DB[Database Servers<br/>10.0.20.0/24]
        CACHE[Cache Servers<br/>10.0.20.0/24]
    end
    
    subgraph "Management Zone - VLAN 30"
        MON[Monitoring<br/>10.0.30.0/24]
        LOG[Logging<br/>10.0.30.0/24]
    end
    
    CLIENT --> FW1
    FW1 --> LB
    LB --> APP
    APP --> DB
    APP --> CACHE
    APP --> MON
    APP --> LOG
    
    style FW1 fill:#ff6666
    style APP fill:#66ff66
    style DB fill:#6666ff
    style MON fill:#ffff66
```

---

## Technology Stack

```mermaid
mindmap
  root((QSFL-CAAD))
    Backend
      Python 3.8+
      Flask/FastAPI
      NumPy/SciKit-Learn
      Cryptography
    Frontend
      React/TypeScript
      Vite
      TailwindCSS
      Plotly/D3.js
    Security
      CRYSTALS-Kyber
      CRYSTALS-Dilithium
      TLS 1.3
      JWT
    Data
      PostgreSQL
      Redis
      SQLite
      MinIO
    Monitoring
      Prometheus
      Grafana
      ELK Stack
      Jaeger
    DevOps
      Docker
      Kubernetes
      GitHub Actions
      Terraform
```

---

## Scalability Patterns

### Horizontal Scaling

```mermaid
graph LR
    subgraph "Auto-Scaling Group"
        direction TB
        I1[Instance 1]
        I2[Instance 2]
        I3[Instance 3]
        IN[Instance N]
    end
    
    LB[Load Balancer] --> I1 & I2 & I3 & IN
    
    I1 & I2 & I3 & IN --> SHARED[(Shared State<br/>Redis/DB)]
    
    METRICS[Metrics] --> ASG[Auto-Scaling<br/>Policy]
    ASG --> |Scale Up/Down| I1 & I2 & I3 & IN
    
    style LB fill:#90EE90
    style SHARED fill:#87CEEB
    style ASG fill:#FFB6C1
```

---

## Summary

This architecture provides:

1. **Modularity**: Clear separation of concerns across layers
2. **Scalability**: Horizontal scaling capabilities at each tier
3. **Security**: Defense in depth with multiple security layers
4. **Observability**: Comprehensive monitoring and logging
5. **Resilience**: High availability and fault tolerance
6. **Flexibility**: Support for multiple deployment scenarios

For implementation details, refer to:
- [API Documentation](API_DOCUMENTATION.md)
- [Deployment Guide](DEPLOYMENT_GUIDE.md)
- [Security Guide](SECURITY_GUIDE.md)
