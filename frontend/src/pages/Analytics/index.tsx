import { motion } from 'framer-motion'
import PageHeader from '@/components/ui/PageHeader'

export default function Analytics() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      <PageHeader
        title="Advanced Analytics"
        description="Deep insights and predictive analytics"
      />
      
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4">Analytics Dashboard</h3>
        <p className="text-gray-600 dark:text-gray-400">
          Advanced analytics features coming soon...
        </p>
      </div>
    </motion.div>
  )
}