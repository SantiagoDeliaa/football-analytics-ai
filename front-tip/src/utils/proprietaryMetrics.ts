import type { MatchMetrics } from '../types/eventData'
import { formatMetricValue } from './formatters'

export interface ProprietaryMetricItem {
  title: string
  value: string
  subtitle?: string
  summary: string
  whatItMeasures: string
  interpretation: string
  highMeaning: string
  lowMeaning: string
  limitations: string
  coachQuestion: string
}

function buildMetricStatusLabel(rawLabel?: string) {
  return rawLabel?.trim() || 'Sin referencia'
}

export function buildProprietaryMetricItems(metrics: MatchMetrics): ProprietaryMetricItem[] {
  return [
    {
      title: 'Dominio territorial',
      value: formatMetricValue(metrics.field_tilt_index),
      subtitle: buildMetricStatusLabel(metrics.field_tilt_label),
      summary: 'Mide cuánto juega el equipo en zonas avanzadas del campo rival.',
      whatItMeasures: 'Compara las acciones del equipo en el último tercio respecto del total del partido.',
      interpretation: 'Sirve para leer control territorial y capacidad para instalarse cerca del arco rival.',
      highMeaning: 'Un valor alto sugiere dominio territorial y presencia sostenida en campo rival.',
      lowMeaning: 'Un valor bajo sugiere menor presencia ofensiva o más tiempo defendiendo lejos del arco rival.',
      limitations: 'Necesita coordenadas de eventos. Sin selección de equipo no aplica y algunos providers pueden tener menos precisión espacial.',
      coachQuestion: '¿Qué significa el dominio territorial de este equipo en este partido?',
    },
    {
      title: 'Verticalidad',
      value: formatMetricValue(metrics.directness_index),
      subtitle: buildMetricStatusLabel(metrics.directness_label),
      summary: 'Mide qué tan directo progresa el equipo cuando tiene la pelota.',
      whatItMeasures: 'Relaciona acciones progresivas con el volumen de pases del recorte analizado.',
      interpretation: 'Ayuda a distinguir ataques rápidos y profundos frente a posesiones más pausadas.',
      highMeaning: 'Un valor alto indica intención de avanzar rápido y atacar hacia adelante con frecuencia.',
      lowMeaning: 'Un valor bajo indica circulación más paciente o poca capacidad para romper líneas.',
      limitations: 'Depende de cómo el provider identifique progresión; no describe por sí sola la calidad de la posesión.',
      coachQuestion: '¿Cómo interpreto la verticalidad del equipo en este partido?',
    },
    {
      title: 'Amenaza progresiva',
      value: formatMetricValue(metrics.progressive_threat_index),
      subtitle: buildMetricStatusLabel(metrics.progressive_threat_label),
      summary: 'Resume cuánta amenaza ofensiva genera el equipo al avanzar.',
      whatItMeasures: 'Combina acciones progresivas, presencia en último tercio, remates y volumen de xG.',
      interpretation: 'Es útil para detectar si el avance del equipo realmente termina en peligro.',
      highMeaning: 'Un valor alto sugiere progresión con profundidad y capacidad real de generar ocasiones.',
      lowMeaning: 'Un valor bajo sugiere que el equipo avanza poco o que sus avances no terminan en peligro claro.',
      limitations: 'Mejora cuando hay xG y coordenadas; con providers limitados puede reflejar más volumen que calidad fina.',
      coachQuestion: '¿Qué dice la amenaza progresiva sobre la producción ofensiva del equipo?',
    },
    {
      title: 'Altura de recuperación',
      value: formatMetricValue(metrics.recovery_height_index),
      subtitle: buildMetricStatusLabel(metrics.recovery_height_label),
      summary: 'Mide dónde recupera la pelota el equipo sobre el campo.',
      whatItMeasures: 'Promedia la zona de recuperación a partir de acciones defensivas con ubicación registrada.',
      interpretation: 'Ayuda a leer si el equipo presiona alto o si recupera más cerca de su propio arco.',
      highMeaning: 'Un valor alto indica recuperaciones adelantadas y presión más agresiva.',
      lowMeaning: 'Un valor bajo indica recuperaciones más retrasadas o un bloque defensivo más bajo.',
      limitations: 'Depende de eventos defensivos con coordenadas; si faltan datos espaciales puede no aplicar.',
      coachQuestion: '¿Qué me dice la altura de recuperación sobre la presión del equipo?',
    },
    {
      title: 'Calidad de remate',
      value: formatMetricValue(metrics.shot_quality_index),
      subtitle: buildMetricStatusLabel(metrics.shot_quality_label),
      summary: 'Mide la calidad promedio de las ocasiones finalizadas en remate.',
      whatItMeasures: 'Se basa en la relación entre el xG total y la cantidad de remates del recorte.',
      interpretation: 'Permite distinguir entre volumen de tiro y calidad real de las situaciones generadas.',
      highMeaning: 'Un valor alto indica remates desde zonas o contextos de mayor probabilidad de gol.',
      lowMeaning: 'Un valor bajo indica remates lejanos, forzados o de baja calidad.',
      limitations: 'Sin xG confiable se vuelve una aproximación. Algunos providers pueden subestimar la calidad real del remate.',
      coachQuestion: '¿Qué dice la calidad de remate sobre la producción ofensiva del equipo?',
    },
    {
      title: 'Influencia del jugador',
      value: formatMetricValue(metrics.player_influence_score),
      subtitle: buildMetricStatusLabel(metrics.player_influence_label),
      summary: 'Resume el peso del jugador seleccionado en el juego del equipo.',
      whatItMeasures: 'Combina participación total, progresión, presencia ofensiva y aportes defensivos del jugador.',
      interpretation: 'Sirve para ver si un futbolista impacta en varias fases del partido y no sólo en una acción puntual.',
      highMeaning: 'Un valor alto indica protagonismo sostenido y participación influyente en distintas fases.',
      lowMeaning: 'Un valor bajo indica menor participación o intervención más aislada en el recorte.',
      limitations: 'Sólo aplica cuando se selecciona un jugador específico y depende del volumen de eventos atribuibles por el provider.',
      coachQuestion: '¿Cómo debería interpretar la influencia de este jugador en el partido?',
    },
  ]
}
