import { FileText, Calendar, Scale, TrendingUp } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";

interface SimilarCaseCardProps {
  title: string;
  court: string;
  year: number;
  confidence: number;
  summary?: string;
  onClick: () => void;
}

export const SimilarCaseCard = ({
  title,
  court,
  year,
  confidence,
  summary,
  onClick,
}: SimilarCaseCardProps) => {
  return (
    <Card
      onClick={onClick}
      className="p-6 hover:shadow-strong transition-smooth cursor-pointer border border-border hover:border-accent/50 scale-in group"
    >
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="p-2 legal-gradient rounded-lg shadow-medium group-hover:shadow-gold transition-smooth">
            <FileText className="h-5 w-5 text-primary-foreground" />
          </div>
          <div>
            <h3 className="font-semibold text-foreground group-hover:text-primary transition-smooth">
              {title}
            </h3>
            <div className="flex items-center gap-4 mt-1">
              <div className="flex items-center gap-1 text-xs text-muted-foreground">
                <Scale className="h-3 w-3" />
                {court}
              </div>
              <div className="flex items-center gap-1 text-xs text-muted-foreground">
                <Calendar className="h-3 w-3" />
                {year}
              </div>
            </div>
          </div>
        </div>
      </div>

      {summary && (
        <p className="text-sm text-muted-foreground mb-4 line-clamp-2">
          {summary}
        </p>
      )}

      <div className="space-y-2">
        <div className="flex items-center justify-between text-xs">
          <span className="flex items-center gap-1 text-muted-foreground">
            <TrendingUp className="h-3 w-3" />
            Similarity Score
          </span>
          <span className="font-semibold text-accent">{confidence}%</span>
        </div>
        <Progress value={confidence} className="h-2" />
      </div>
    </Card>
  );
};
