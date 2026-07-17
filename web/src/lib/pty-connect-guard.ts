interface OpenPtySocketOptions<Socket> {
  buildUrl: () => Promise<string>;
  isCurrent: () => boolean;
  createSocket: (url: string) => Socket;
}

export async function openPtySocket<Socket>({
  buildUrl,
  isCurrent,
  createSocket,
}: OpenPtySocketOptions<Socket>): Promise<Socket | null> {
  const url = await buildUrl();
  if (!isCurrent()) return null;
  return createSocket(url);
}
