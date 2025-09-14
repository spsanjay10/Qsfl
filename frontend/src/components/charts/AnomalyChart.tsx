import { useMemo } from 'react'
import Plot from 'react-plotly.js'

interface AnomalyChartProps {
  data: any
}

export default function AnomalyChart({ data }: AnomalyChartProps) {
  const plotData = useMemo(() => {
    if (!data?.metrics?.anomaly_scores) return []

    const traces = []
    const timestamps = data.metrics.timestamps || []

    // Threshold line
    if (timestamps.length > 0) {
      traces.push({
        x: timestamps,
        y: Array(timestamps.length).fill(0.6),
        name: 'Threshold',
        type: 'scatter',
        mode: 'lines',
        line: { color: '#ef4444', dash: 'dash', width: 2 },
        hovertemplate: 'Threshold: 0.6<extra></extra>',
      })
    }

    // Client traces
    Object.entries(data.metrics.anomaly_scores || {}).forEach(([clientId, scores]: [string, any]) => {
      if (scores && scores.length > 0) {
        const client = data.clients?.[clientId] || {}
        const getClientColor = (type: string) => {
          switch (type) {
            case 'honest': return '#22c55e'
            case 'suspicious': return '#f59e0b'
            case 'malicious': return '#ef4444'
            default: return '#6b7280'
          }
        }
        
        const color = getClientColor(client.type)
        const clientTimestamps = timestamps.slice(-scores.length)

        traces.push({
          x: clientTimestamps,
          y: scores,
          name: clientId,
          type: 'scatter',
          mode: 'lines+markers',
          line: { color, width: 2 },
          marker: { size: 6, color },
          opacity: client.quarantined ? 0.5 : 1.0,
          hovertemplate: `${clientId}<br>Anomaly Score: %{y:.3f}<br>Time: %{x}<extra></extra>`,
        })
      }
    })

    return traces
  }, [data])

  const layout = {
    title: {
      text: 'Real-time Anomaly Detection',
      font: { size: 18, color: '#374151' },
    },
    xaxis: {
      title: 'Time',
      gridcolor: '#e5e7eb',
      tickformat: '%H:%M:%S',
    },
    yaxis: {
      title: 'Anomaly Score',
      range: [0, 1],
      gridcolor: '#e5e7eb',
    },
    showlegend: true,
    legend: {
      orientation: 'h',
      y: -0.2,
    },
    margin: { t: 60, r: 40, b: 80, l: 60 },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { family: 'Inter, system-ui, sans-serif' },
  }

  const config = {
    responsive: true,
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-soft p-6 border border-gray-200 dark:border-gray-700">
      <Plot
        data={plotData}
        layout={layout}
        config={config}
        className="w-full"
        style={{ height: '400px' }}
      />
    </div>
  )
}