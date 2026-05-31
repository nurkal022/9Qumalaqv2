type Props = { className?: string };

export default function OrnamentBorder({ className = "" }: Props) {
  return (
    <div className={`text-accent-gold/40 ${className}`}>
      <img src="/ornaments/tumarsha.svg" alt="" className="w-full h-3 opacity-60" />
    </div>
  );
}
