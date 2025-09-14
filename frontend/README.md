# QSFL-CAAD Frontend

Modern React frontend for the Quantum-Safe Federated Learning with Comprehensive Anomaly and Attack Detection system.

## 🚀 Features

- **Modern React 18** with TypeScript
- **Vite** for fast development and building
- **Tailwind CSS** for styling
- **Framer Motion** for animations
- **React Query** for data fetching
- **Socket.IO** for real-time updates
- **Plotly.js** for interactive charts
- **Headless UI** for accessible components
- **React Hook Form** with Zod validation
- **Vitest** for testing
- **Storybook** for component development

## 📦 Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## 🛠️ Development

### Available Scripts

```bash
# Development
npm run dev              # Start dev server
npm run build           # Build for production
npm run preview         # Preview production build

# Code Quality
npm run lint            # Run ESLint
npm run lint:fix        # Fix ESLint issues
npm run format          # Format with Prettier
npm run format:check    # Check formatting
npm run type-check      # TypeScript type checking

# Testing
npm run test            # Run tests
npm run test:ui         # Run tests with UI
npm run test:coverage   # Run tests with coverage

# Storybook
npm run storybook       # Start Storybook
npm run build-storybook # Build Storybook

# Analysis
npm run analyze         # Bundle analyzer
```

### Project Structure

```
src/
├── components/         # Reusable components
│   ├── ui/            # Basic UI components
│   ├── charts/        # Chart components
│   └── Layout/        # Layout components
├── pages/             # Page components
├── hooks/             # Custom hooks
├── contexts/          # React contexts
├── api/               # API functions
├── utils/             # Utility functions
├── types/             # TypeScript types
└── assets/            # Static assets
```

## 🎨 Styling

### Tailwind CSS

The project uses Tailwind CSS with a custom design system:

- **Colors**: Primary, secondary, success, warning, danger
- **Typography**: Inter font family
- **Spacing**: Consistent spacing scale
- **Animations**: Custom animations and transitions

### Dark Mode

Dark mode is supported and automatically detects system preference:

```tsx
import { useTheme } from '@/hooks/useTheme'

function Component() {
  const { theme, toggleTheme } = useTheme()
  
  return (
    <button onClick={toggleTheme}>
      {theme === 'dark' ? 'Light' : 'Dark'} Mode
    </button>
  )
}
```

## 📊 Data Management

### React Query

Used for server state management:

```tsx
import { useQuery } from '@tanstack/react-query'
import { getDashboardData } from '@/api/dashboard'

function Dashboard() {
  const { data, isLoading, error } = useQuery({
    queryKey: ['dashboard'],
    queryFn: getDashboardData,
    refetchInterval: 5000,
  })
  
  if (isLoading) return <LoadingSpinner />
  if (error) return <ErrorMessage error={error} />
  
  return <DashboardContent data={data} />
}
```

### Socket.IO

Real-time updates via WebSocket:

```tsx
import { useSocket } from '@/hooks/useSocket'

function RealTimeComponent() {
  const { data, emit } = useSocket('dashboard_update')
  
  const handleAction = () => {
    emit('client_action', { action: 'start' })
  }
  
  return <div>{JSON.stringify(data)}</div>
}
```

## 🧪 Testing

### Vitest + Testing Library

```tsx
import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import MetricCard from '@/components/ui/MetricCard'

describe('MetricCard', () => {
  it('renders metric data correctly', () => {
    render(
      <MetricCard
        title="Active Clients"
        value={42}
        change="+5%"
        changeType="positive"
      />
    )
    
    expect(screen.getByText('Active Clients')).toBeInTheDocument()
    expect(screen.getByText('42')).toBeInTheDocument()
    expect(screen.getByText('+5%')).toBeInTheDocument()
  })
})
```

### Coverage

Run tests with coverage:

```bash
npm run test:coverage
```

Coverage reports are generated in `coverage/` directory.

## 📱 Responsive Design

The application is fully responsive:

- **Mobile**: Collapsible sidebar, touch-friendly controls
- **Tablet**: Optimized layout for medium screens
- **Desktop**: Full sidebar, multi-column layouts

## 🔧 Configuration

### Environment Variables

Create `.env.local` file:

```env
VITE_API_URL=http://localhost:5000
VITE_WS_URL=ws://localhost:5000
```

### Vite Configuration

Key configurations in `vite.config.ts`:

- **Path aliases** for clean imports
- **Proxy setup** for API calls
- **Build optimization** with code splitting
- **Test configuration** with Vitest

## 🚀 Deployment

### Build for Production

```bash
npm run build
```

The build output will be in the `dist/` directory.

### Docker

```dockerfile
FROM node:18-alpine as builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### Environment-specific Builds

```bash
# Development
npm run build -- --mode development

# Staging
npm run build -- --mode staging

# Production
npm run build -- --mode production
```

## 🎯 Performance

### Optimization Techniques

- **Code splitting** with dynamic imports
- **Tree shaking** for smaller bundles
- **Image optimization** with modern formats
- **Lazy loading** for components and routes
- **Memoization** with React.memo and useMemo

### Bundle Analysis

```bash
npm run analyze
```

This opens a visual representation of your bundle size.

## 🔒 Security

### Best Practices

- **Input validation** with Zod schemas
- **XSS protection** with proper escaping
- **CSRF protection** with tokens
- **Content Security Policy** headers
- **Secure headers** configuration

## 🤝 Contributing

### Code Style

- Use **TypeScript** for all new code
- Follow **ESLint** and **Prettier** configurations
- Write **tests** for new components
- Use **semantic commit messages**

### Component Development

1. Create component in appropriate directory
2. Add TypeScript interfaces
3. Write Storybook stories
4. Add unit tests
5. Update documentation

### Pull Request Process

1. Create feature branch
2. Make changes with tests
3. Run quality checks: `npm run lint && npm run type-check && npm run test`
4. Submit pull request

## 📚 Resources

- [React Documentation](https://react.dev/)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
- [Tailwind CSS](https://tailwindcss.com/docs)
- [Framer Motion](https://www.framer.com/motion/)
- [React Query](https://tanstack.com/query/latest)
- [Vitest](https://vitest.dev/)
- [Storybook](https://storybook.js.org/)

## 🐛 Troubleshooting

### Common Issues

**Build fails with TypeScript errors:**
```bash
npm run type-check
```

**Tests failing:**
```bash
npm run test -- --reporter=verbose
```

**Styling issues:**
```bash
# Rebuild Tailwind
npm run build:css
```

**Socket connection issues:**
- Check backend server is running
- Verify VITE_API_URL environment variable
- Check browser console for WebSocket errors

### Debug Mode

Enable debug logging:

```env
VITE_DEBUG=true
```

This enables additional console logging for debugging.

---

For more information, see the main project documentation.