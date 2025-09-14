import { motion } from 'framer-motion'
import { ArrowUpIcon, ArrowDownIcon } from '@heroicons/react/24/solid'
import { cn } from '@/utils/cn'

interface MetricCardProps {
  title: string
  value: string | number
  change?: string
  changeType?: 'positive' | 'negative' | 'neutral'
  icon?: React.ComponentType<{ className?: string }>
  loading?: boolean
}

export default function MetricCard({
  title,
  value,
  change,
  changeType = 'neutral',
  icon: Icon,
  loading = false,
}: MetricCardProps) {
  if (loading) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-soft p-6 animate-pulse">
        <div className="flex items-center justify-between">
          <div className="space-y-2">
            <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-20"></div>
            <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded w-16"></div>
          </div>
          <div className="h-8 w-8 bg-gray-200 dark:bg-gray-700 rounded"></div>
        </div>
        <div className="mt-4 h-4 bg-gray-200 dark:bg-gray-700 rounded w-24"></div>
      </div>
    )
  }

  const changeColor = {
    positive: 'text-success-600 dark:text-success-400',
    negative: 'text-danger-600 dark:text-danger-400',
    neutral: 'text-gray-600 dark:text-gray-400',
  }[changeType]

  const changeIcon = changeType === 'positive' ? ArrowUpIcon : changeType === 'negative' ? ArrowDownIcon : null

  return (
    <motion.div
      whileHover={{ y: -2 }}
      className="bg-white dark:bg-gray-800 rounded-lg shadow-soft hover:shadow-medium transition-all duration-200 p-6 border border-gray-100 dark:border-gray-700"
    >
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <p className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">
            {title}
          </p>
          <p className="text-2xl font-bold text-gray-900 dark:text-white">
            {typeof value === 'number' ? value.toLocaleString() : value}
          </p>
        </div>
        {Icon && (
          <div className="flex-shrink-0">
            <div className="p-3 bg-gradient-to-br from-primary-500 to-secondary-500 rounded-lg">
              <Icon className="h-6 w-6 text-white" />
            </div>
          </div>
        )}
      </div>
      
      {change && (
        <div className="mt-4 flex items-center">
          {changeIcon && (
            <changeIcon className={cn('h-4 w-4 mr-1', changeColor)} />
          )}
          <span className={cn('text-sm font-medium', changeColor)}>
            {change}
          </span>
        </div>
      )}
    </motion.div>
  )
}