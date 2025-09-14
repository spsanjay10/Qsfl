import { motion } from 'framer-motion'
import { useQuery } from '@tanstack/react-query'

import PageHeader from '@/components/ui/PageHeader'
import MetricCard from '@/components/ui/MetricCard'
import AnomalyChart from '@/components/charts/AnomalyChart'
import ClientMap from '@/components/charts/ClientMap'
import SecurityEvents from '@/components/SecurityEvents'
import SystemControls from '@/components/SystemControls'
import ConnectedClients from '@/components/ConnectedClients'

import { useSocket } from '@/hooks/useSocket'
import { getDashboardData } from '@/api/dashboard'

import {
  UsersIcon,
  ShieldCheckIcon,
  ChartBarIcon,
  CpuChipIcon,
} from '@heroicons/react/24/outline'

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
    },
  },
}

const itemVariants = {
  hidden: { y: 20, opacity: 0 },
  visible: {
    y: 0,
    opacity: 1,
    transition: {
      type: 'spring',
      stiffness: 100,
    },
  },
}

export default function Dashboard() {
  const { data: dashboardData, isLoading } = useQuery({
    queryKey: ['dashboard'],
    queryFn: getDashboardData,
    refetchInterval: 5000, // Refetch every 5 seconds
  })

  const { data: realtimeData } = useSocket('dashboard_update')

  // Use real-time data if available, otherwise use query data
  const data = realtimeData || dashboardData

  const metrics = [
    {
      title: 'Active Clients',
      value: data?.clients ? Object.values(data.clients).filter((c: any) => c.status === 'active' && !c.quarantined).length : 0,
      change: '+2.5%',
      changeType: 'positive' as const,
      icon: UsersIcon,
    },
    {
      title: 'System Status',
      value: data?.system_status?.toUpperCase() || 'STOPPED',
      change: data?.system_status === 'running' ? 'Online' : 'Offline',
      changeType: data?.system_status === 'running' ? 'positive' : 'negative' as const,
      icon: CpuChipIcon,
    },
    {
      title: 'Training Round',
      value: data?.current_round || 0,
      change: 'Round in progress',
      changeType: 'neutral' as const,
      icon: ChartBarIcon,
    },
    {
      title: 'Security Score',
      value: '98.5%',
      change: '+0.3%',
      changeType: 'positive' as const,
      icon: ShieldCheckIcon,
    },
  ]

  if (isLoading) {
    return (
      <div className="animate-pulse">
        <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded w-1/4 mb-6"></div>
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4 mb-8">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="bg-white dark:bg-gray-800 rounded-lg p-6 h-32"></div>
          ))}
        </div>
      </div>
    )
  }

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="space-y-6"
    >
      <motion.div variants={itemVariants}>
        <PageHeader
          title="Dashboard"
          description="Monitor your federated learning system in real-time"
        >
          <SystemControls />
        </PageHeader>
      </motion.div>

      {/* Metrics */}
      <motion.div
        variants={itemVariants}
        className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4"
      >
        {metrics.map((metric, index) => (
          <motion.div
            key={metric.title}
            variants={itemVariants}
            custom={index}
          >
            <MetricCard {...metric} />
          </motion.div>
        ))}
      </motion.div>

      {/* Charts */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <motion.div variants={itemVariants} className="lg:col-span-2">
          <AnomalyChart data={data} />
        </motion.div>
        <motion.div variants={itemVariants}>
          <ClientMap data={data} />
        </motion.div>
      </div>

      {/* Connected Clients Section */}
      <motion.div variants={itemVariants}>
        <ConnectedClients data={data} />
      </motion.div>

      {/* Security Events */}
      <motion.div variants={itemVariants}>
        <SecurityEvents data={data} />
      </motion.div>
    </motion.div>
  )
}