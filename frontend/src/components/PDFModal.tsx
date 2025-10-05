import { X, FileText, ThumbsUp, ThumbsDown } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

interface PDFModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  caseTitle: string;
  citations?: string[];
  sentiment?: "positive" | "negative" | "neutral";
}

export const PDFModal = ({
  open,
  onOpenChange,
  caseTitle,
  citations = [],
  sentiment = "neutral",
}: PDFModalProps) => {
  const sentimentConfig = {
    positive: {
      icon: ThumbsUp,
      text: "Favorable Judgment",
      color: "text-green-600",
      bg: "bg-green-100",
    },
    negative: {
      icon: ThumbsDown,
      text: "Unfavorable Judgment",
      color: "text-red-600",
      bg: "bg-red-100",
    },
    neutral: {
      icon: FileText,
      text: "Neutral Judgment",
      color: "text-muted-foreground",
      bg: "bg-muted",
    },
  };

  const config = sentimentConfig[sentiment];
  const SentimentIcon = config.icon;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-5xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="text-2xl flex items-center justify-between">
            <span>{caseTitle}</span>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => onOpenChange(false)}
            >
              <X className="h-5 w-5" />
            </Button>
          </DialogTitle>
        </DialogHeader>

        <div className="space-y-6 mt-4">
          {/* Sentiment Analysis */}
          <div className={`p-4 rounded-lg ${config.bg} flex items-center gap-3`}>
            <SentimentIcon className={`h-6 w-6 ${config.color}`} />
            <div>
              <h4 className={`font-semibold ${config.color}`}>
                Sentiment Analysis
              </h4>
              <p className="text-sm text-muted-foreground">{config.text}</p>
            </div>
          </div>

          {/* Citations Section */}
          {citations.length > 0 && (
            <div>
              <h4 className="font-semibold text-foreground mb-3">
                Key Citations
              </h4>
              <div className="space-y-2">
                {citations.map((citation, index) => (
                  <div
                    key={index}
                    className="p-3 bg-accent/10 border border-accent/30 rounded-lg"
                  >
                    <Badge variant="outline" className="mb-2">
                      Citation {index + 1}
                    </Badge>
                    <p className="text-sm text-foreground">{citation}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* PDF Viewer Placeholder */}
          <div className="border-2 border-dashed border-border rounded-lg p-12 bg-muted/30 text-center">
            <FileText className="h-16 w-16 mx-auto mb-4 text-muted-foreground" />
            <h4 className="font-semibold text-foreground mb-2">
              PDF Document Preview
            </h4>
            <p className="text-sm text-muted-foreground">
              Document viewer with highlighted citations would appear here
            </p>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};
