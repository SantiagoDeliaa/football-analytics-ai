interface TabsProps<T extends string> {
  value: T
  onChange: (value: T) => void
  options: { value: T; label: string }[]
}

export function Tabs<T extends string>({ value, onChange, options }: TabsProps<T>) {
  return (
    <div className="flex w-full flex-wrap gap-2 rounded-xl border border-slate-700 bg-slate-900/60 p-2">
      {options.map((option) => {
        const active = value === option.value
        return (
          <button
            className={`rounded-lg px-4 py-2 text-sm font-medium transition ${
              active
                ? 'bg-slate-700 text-slate-50'
                : 'bg-transparent text-slate-300 hover:bg-slate-800 hover:text-slate-50'
            }`}
            key={option.value}
            onClick={() => onChange(option.value)}
            type="button"
          >
            {option.label}
          </button>
        )
      })}
    </div>
  )
}
