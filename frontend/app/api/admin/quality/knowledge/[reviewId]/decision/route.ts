import { proxyKnowledgeWrite } from "@/lib/qualityProxy";

export async function POST(request: Request, { params }: { params: Promise<{ reviewId: string }> }) {
  return proxyKnowledgeWrite(request, (await params).reviewId);
}
