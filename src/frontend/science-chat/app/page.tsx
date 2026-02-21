import { ChatWindow } from "@/components/chat/ChatWindow";
import { Sidebar } from "@/components/layout/Sidebar";

export default function Home() {
  return (
    <main className="flex h-screen bg-slate-950 overflow-hidden">
      <Sidebar />
      <div className="flex-1 flex flex-col p-4 min-w-0">
        <ChatWindow />
      </div>
    </main>
  );
}