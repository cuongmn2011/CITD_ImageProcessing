import type { PlateResult } from "../types";

interface PlateTableProps {
  plates: PlateResult[];
}

export function PlateTable({ plates }: PlateTableProps) {
  return (
    <section className="panel table-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">TRACKS</p>
          <h2>Biển số trong khung hình</h2>
        </div>
        <span className="count-pill">{plates.length}</span>
      </div>
      {plates.length === 0 ? (
        <p className="empty-state">Chưa phát hiện biển số trong frame hiện tại.</p>
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Track</th>
                <th>Biển số</th>
                <th>Trạng thái</th>
                <th>Detect</th>
                <th>OCR</th>
              </tr>
            </thead>
            <tbody>
              {plates.map((plate, index) => (
                <tr key={`${plate.track_id ?? "untracked"}-${index}`}>
                  <td>#{plate.track_id ?? "—"}</td>
                  <td className="plate-text">{plate.text || "—"}</td>
                  <td><span className={`status status-${plate.status}`}>{plate.status}</span></td>
                  <td>{(plate.detection_confidence * 100).toFixed(0)}%</td>
                  <td>{plate.ocr_confidence ? `${(plate.ocr_confidence * 100).toFixed(0)}%` : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
