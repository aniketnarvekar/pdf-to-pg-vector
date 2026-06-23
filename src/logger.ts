
import winston from 'winston';

// Define layout rules for formatting logs
const logFormat = winston.format.combine(
  winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
  winston.format.errors({ stack: true }), // Automatically extracts full Error stack traces
  winston.format.json() // Exports logs in strict JSON formatting for engines
);

// Instantiate your global logger instance
export const logger = winston.createLogger({
  level: process.env.NODE_ENV === 'production' ? 'info' : 'debug',
  format: logFormat,
  defaultMeta: { service: 'api-gateway' }, // Tags every log entry with service metadata
  transports: [
    new winston.transports.Console({
      // Switch back to readable, colorized text outputs during local terminal sessions
      format: process.env.NODE_ENV !== 'production'
        ? winston.format.combine(winston.format.colorize(), winston.format.simple())
        : logFormat
    })
  ]
});
