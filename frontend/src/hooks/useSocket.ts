import { useContext, useEffect, useState } from 'react'
import { SocketContext } from '@/contexts/SocketContext'

export function useSocket(event?: string) {
  const { socket, isConnected } = useContext(SocketContext)
  const [data, setData] = useState<any>(null)

  useEffect(() => {
    if (!socket || !event) return

    const handleData = (newData: any) => {
      setData(newData)
    }

    socket.on(event, handleData)

    return () => {
      socket.off(event, handleData)
    }
  }, [socket, event])

  const emit = (eventName: string, data?: any) => {
    if (socket) {
      socket.emit(eventName, data)
    }
  }

  return {
    socket,
    isConnected,
    data,
    emit,
  }
}