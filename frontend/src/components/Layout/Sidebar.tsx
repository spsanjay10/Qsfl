import { NavLink } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  HomeIcon,
  UsersIcon,
  ShieldCheckIcon,
  ChartBarIcon,
  CogIcon,
} from '@heroicons/react/24/outline'

import { cn } from '@/utils/cn'

const navigation = [
  { name: 'Dashboard', href: '/dashboard', icon: HomeIcon },
  { name: 'Clients', href: '/clients', icon: UsersIcon },
  { name: 'Security', href: '/security', icon: ShieldCheckIcon },
  { name: 'Analytics', href: '/analytics', icon: ChartBarIcon },
  { name: 'Settings', href: '/settings', icon: CogIcon },
]

export default function Sidebar() {
  return (
    <div className="flex grow flex-col gap-y-5 overflow-y-auto bg-white dark:bg-gray-800 px-6 pb-4 ring-1 ring-gray-900/5 dark:ring-gray-100/5">
      {/* Logo */}
      <div className="flex h-16 shrink-0 items-center">
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          className="flex items-center space-x-3"
        >
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-primary-500 to-secondary-500">
            <ShieldCheckIcon className="h-5 w-5 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-gray-900 dark:text-white">
              QSFL-CAAD
            </h1>
            <p className="text-xs text-gray-500 dark:text-gray-400">
              Quantum-Safe FL
            </p>
          </div>
        </motion.div>
      </div>

      {/* Navigation */}
      <nav className="flex flex-1 flex-col">
        <ul role="list" className="flex flex-1 flex-col gap-y-7">
          <li>
            <ul role="list" className="-mx-2 space-y-1">
              {navigation.map((item, index) => (
                <motion.li
                  key={item.name}
                  initial={{ x: -20, opacity: 0 }}
                  animate={{ x: 0, opacity: 1 }}
                  transition={{ delay: index * 0.1 }}
                >
                  <NavLink
                    to={item.href}
                    className={({ isActive }) =>
                      cn(
                        'group flex gap-x-3 rounded-md p-2 text-sm leading-6 font-semibold transition-all duration-200',
                        isActive
                          ? 'bg-gradient-to-r from-primary-500 to-secondary-500 text-white shadow-md'
                          : 'text-gray-700 dark:text-gray-200 hover:text-primary-600 dark:hover:text-primary-400 hover:bg-gray-50 dark:hover:bg-gray-700'
                      )
                    }
                  >
                    {({ isActive }) => (
                      <>
                        <item.icon
                          className={cn(
                            'h-6 w-6 shrink-0 transition-colors',
                            isActive
                              ? 'text-white'
                              : 'text-gray-400 group-hover:text-primary-600 dark:group-hover:text-primary-400'
                          )}
                        />
                        {item.name}
                        {isActive && (
                          <motion.div
                            layoutId="activeTab"
                            className="absolute right-0 top-0 bottom-0 w-1 bg-white rounded-l-full"
                          />
                        )}
                      </>
                    )}
                  </NavLink>
                </motion.li>
              ))}
            </ul>
          </li>

          {/* System Status */}
          <li className="mt-auto">
            <div className="rounded-lg bg-gray-50 dark:bg-gray-700/50 p-4">
              <div className="flex items-center space-x-3">
                <div className="flex-shrink-0">
                  <div className="h-3 w-3 bg-green-400 rounded-full animate-pulse" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-900 dark:text-white">
                    System Online
                  </p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    All services running
                  </p>
                </div>
              </div>
            </div>
          </li>
        </ul>
      </nav>
    </div>
  )
}