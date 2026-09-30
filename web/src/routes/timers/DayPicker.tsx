import { Checkbox } from "@/components/ui/checkbox"
import { Label } from "@/components/ui/label"
import { WEEKDAYS, type Weekday } from "@/lib/timers"

const SHORT: Record<Weekday, string> = {
  mon: "Mon",
  tue: "Tue",
  wed: "Wed",
  thu: "Thu",
  fri: "Fri",
  sat: "Sat",
  sun: "Sun",
}

/** Seven weekday checkboxes. An empty selection means "once". */
export function DayPicker({
  idPrefix,
  value,
  onChange,
  disabled,
}: {
  idPrefix: string
  value: Weekday[]
  onChange: (next: Weekday[]) => void
  disabled?: boolean
}) {
  const toggle = (day: Weekday, checked: boolean) => {
    const next = new Set(value)
    if (checked) next.add(day)
    else next.delete(day)
    onChange(WEEKDAYS.filter((candidate) => next.has(candidate)))
  }

  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1">
      {WEEKDAYS.map((day) => (
        <div key={day} className="touch-target flex items-center gap-2">
          <Checkbox
            id={`${idPrefix}-${day}`}
            checked={value.includes(day)}
            disabled={disabled}
            onCheckedChange={(checked) => toggle(day, checked === true)}
          />
          <Label htmlFor={`${idPrefix}-${day}`}>{SHORT[day]}</Label>
        </div>
      ))}
    </div>
  )
}
