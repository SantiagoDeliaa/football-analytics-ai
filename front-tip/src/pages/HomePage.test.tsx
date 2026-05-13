import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { HomePage } from './HomePage'

describe('HomePage', () => {
  it('muestra los módulos principales', () => {
    render(
      <MemoryRouter>
        <HomePage />
      </MemoryRouter>,
    )

    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent(
      /tactical intelligence platform/i,
    )
    expect(screen.getByText('Computer Vision')).toBeInTheDocument()
    expect(screen.getByText('Data Analytics')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: /computer vision/i })).toHaveAttribute(
      'href',
      '/vertical1',
    )
    expect(screen.getByRole('link', { name: /data analytics/i })).toHaveAttribute(
      'href',
      '/vertical2',
    )
  })
})
