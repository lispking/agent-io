//! Ring buffer for short-term memory

use std::collections::VecDeque;

/// A fixed-size ring buffer for storing recent items
#[derive(Debug, Clone)]
pub struct RingBuffer<T> {
    buffer: VecDeque<T>,
    capacity: usize,
}

impl<T> RingBuffer<T> {
    /// Create a new ring buffer with the given capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity: capacity.max(1),
        }
    }

    /// Push an item to the buffer, removing the oldest if at capacity
    pub fn push(&mut self, item: T) {
        if self.buffer.len() == self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(item);
    }

    /// Pop the most recent item
    pub fn pop(&mut self) -> Option<T> {
        self.buffer.pop_back()
    }

    /// Get the number of items in the buffer
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Check if the buffer is full
    pub fn is_full(&self) -> bool {
        self.buffer.len() == self.capacity
    }

    /// Get the capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Iterate over items (most recent last)
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.buffer.iter()
    }

    /// Iterate over items (most recent first)
    pub fn iter_recent(&self) -> impl Iterator<Item = &T> {
        self.buffer.iter().rev()
    }

    /// Get the most recent item
    pub fn last(&self) -> Option<&T> {
        self.buffer.back()
    }

    /// Get the oldest item
    pub fn first(&self) -> Option<&T> {
        self.buffer.front()
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    /// Get all items as a vector (oldest first)
    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.buffer.iter().cloned().collect()
    }

    /// Get all items as a vector (most recent first)
    pub fn to_vec_recent(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.buffer.iter().rev().cloned().collect()
    }
}

impl<T: Clone> RingBuffer<T> {
    /// Create from an existing vector
    pub fn from_vec(items: Vec<T>, capacity: usize) -> Self {
        let capacity = capacity.max(1);
        let mut buffer = VecDeque::with_capacity(capacity);

        // Take only the most recent `capacity` items
        let start = items.len().saturating_sub(capacity);
        for item in items.into_iter().skip(start) {
            if buffer.len() < capacity {
                buffer.push_back(item);
            }
        }

        Self { buffer, capacity }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer_basic() {
        let mut buf: RingBuffer<i32> = RingBuffer::new(3);

        buf.push(1);
        buf.push(2);
        buf.push(3);

        assert_eq!(buf.len(), 3);
        assert_eq!(buf.to_vec(), vec![1, 2, 3]);
    }

    #[test]
    fn test_ring_buffer_overflow() {
        let mut buf: RingBuffer<i32> = RingBuffer::new(3);

        buf.push(1);
        buf.push(2);
        buf.push(3);
        buf.push(4); // Should push out 1

        assert_eq!(buf.len(), 3);
        assert_eq!(buf.to_vec(), vec![2, 3, 4]);
    }

    #[test]
    fn test_ring_buffer_iter_recent() {
        let mut buf: RingBuffer<i32> = RingBuffer::new(3);

        buf.push(1);
        buf.push(2);
        buf.push(3);

        let recent: Vec<_> = buf.iter_recent().copied().collect();
        assert_eq!(recent, vec![3, 2, 1]);
    }

    #[test]
    fn test_ring_buffer_pop() {
        let mut buf: RingBuffer<i32> = RingBuffer::new(3);

        buf.push(1);
        buf.push(2);

        assert_eq!(buf.pop(), Some(2));
        assert_eq!(buf.len(), 1);
    }
}
