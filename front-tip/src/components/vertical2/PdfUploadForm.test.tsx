import { fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { PdfUploadForm } from './PdfUploadForm'

describe('PdfUploadForm', () => {
  it('envía archivo válido', async () => {
    const onSubmit = vi.fn()
    render(<PdfUploadForm loading={false} onSubmit={onSubmit} />)

    const file = new File(['fake pdf'], 'report.pdf', { type: 'application/pdf' })
    const input = screen.getByLabelText(/archivo pdf/i) as HTMLInputElement
    await userEvent.upload(input, file)
    fireEvent.click(screen.getByRole('button', { name: /procesar pdf/i }))

    expect(onSubmit).toHaveBeenCalledWith(file)
  })

  it('muestra validacion inline si el archivo no es PDF', async () => {
    const onSubmit = vi.fn()
    render(<PdfUploadForm loading={false} onSubmit={onSubmit} />)

    const file = new File(['texto'], 'notes.txt', { type: 'text/plain' })
    const input = screen.getByLabelText(/archivo pdf/i) as HTMLInputElement
    await userEvent.upload(input, file)
    fireEvent.click(screen.getByRole('button', { name: /procesar pdf/i }))

    expect(await screen.findByText(/debés seleccionar un archivo pdf/i)).toBeInTheDocument()
    expect(onSubmit).not.toHaveBeenCalled()
  })
})
