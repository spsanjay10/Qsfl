import { motion } from 'framer-motion'
import PageHeader from '@/components/ui/PageHeader'

export default function Settings() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      <PageHeader
        title="Settings"
        description="Configure system parameters and preferences"
      />
      
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4">System Configuration</h3>
        <p className="text-gray-600 dark:text-gray-400">
          Settings panel coming soon...
        </p>
      </div>
    </motion.div>
  )
}