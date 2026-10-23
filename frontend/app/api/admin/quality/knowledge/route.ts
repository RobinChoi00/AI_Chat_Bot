import { proxyKnowledgeWrite } from "@/lib/qualityProxy";

export async function POST(request: Request) {
  return proxyKnowledgeWrite(request);
}
