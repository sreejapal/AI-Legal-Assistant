import { Upload, FileText, X } from "lucide-react";
import { useState, useCallback } from "react";
import { Button } from "@/components/ui/button";

interface FileUploadProps {
  onFileSelect: (file: File | null) => void;
  acceptedFormats?: string;
}

export const FileUpload = ({ onFileSelect, acceptedFormats = ".pdf,.docx" }: FileUploadProps) => {
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) {
      setFile(droppedFile);
      onFileSelect(droppedFile);
    }
  }, [onFileSelect]);

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      onFileSelect(selectedFile);
    }
  };

  const removeFile = () => {
    setFile(null);
    onFileSelect(null);
  };

  return (
    <div className="w-full">
      {!file ? (
        <div
          onDrop={handleDrop}
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={() => setIsDragging(false)}
          className={`border-2 border-dashed rounded-xl p-12 text-center transition-smooth ${
            isDragging
              ? "border-accent bg-accent/10 shadow-gold"
              : "border-border bg-muted/30 hover:border-accent/50 hover:bg-muted/50"
          }`}
        >
          <Upload className="h-16 w-16 mx-auto mb-4 text-accent" />
          <h3 className="text-lg font-semibold mb-2 text-foreground">
            Upload Legal Document
          </h3>
          <p className="text-sm text-muted-foreground mb-6">
            Drag and drop your case file here, or click to browse
          </p>
          <label>
            <Button variant="gold" size="lg" asChild>
              <span className="cursor-pointer">
                Choose File
                <input
                  type="file"
                  accept={acceptedFormats}
                  onChange={handleFileInput}
                  className="hidden"
                />
              </span>
            </Button>
          </label>
          <p className="text-xs text-muted-foreground mt-4">
            Supported formats: PDF, DOCX (Max 10MB)
          </p>
        </div>
      ) : (
        <div className="border border-border rounded-xl p-6 bg-card shadow-medium scale-in">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="p-3 bg-accent/10 rounded-lg">
                <FileText className="h-8 w-8 text-accent" />
              </div>
              <div>
                <h4 className="font-semibold text-foreground">{file.name}</h4>
                <p className="text-sm text-muted-foreground">
                  {(file.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
            </div>
            <Button
              variant="ghost"
              size="icon"
              onClick={removeFile}
              className="hover:bg-destructive/10 hover:text-destructive"
            >
              <X className="h-5 w-5" />
            </Button>
          </div>
        </div>
      )}
    </div>
  );
};
