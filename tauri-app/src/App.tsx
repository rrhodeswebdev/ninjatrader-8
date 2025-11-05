import { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import {
    Play,
    Square,
    Activity,
    Server,
    TrendingUp,
    TrendingDown,
    Minus,
    Brain,
    CheckCircle2,
    XCircle,
    Loader2,
    AlertCircle,
    Moon,
    Sun,
} from 'lucide-react';

interface LogEntry {
    message: string;
    type: 'info' | 'success' | 'error' | 'server';
    timestamp: Date;
}

interface TrainingStatus {
    is_training: boolean;
    progress: string;
    error: string | null;
}

interface HealthStatus {
    status: string;
    model_trained: boolean;
    device: string;
    version?: string;
}

interface LatestSignal {
    signal: 'long' | 'short' | 'hold';
    confidence: number;
    timestamp: Date;
    risk_management?: {
        contracts: number;
        entry_price: number;
        stop_loss: number;
        take_profit: number;
        risk_reward_ratio?: number;
    };
    reasoning?: {
        market_regime?: string;
        filtered?: boolean;
        raw_signal?: string;
        mtf_filtered?: boolean;
        regime_filtered?: boolean;
        confidence_threshold?: number;
    };
}

function App() {
    const [serverRunning, setServerRunning] = useState(false);
    const [logs, setLogs] = useState<LogEntry[]>([]);
    const [loading, setLoading] = useState(false);
    const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(
        null
    );
    const [healthStatus, setHealthStatus] = useState<HealthStatus | null>(null);
    const [latestSignal, setLatestSignal] = useState<LatestSignal | null>(null);
    const [darkMode, setDarkMode] = useState(true); // Default to dark mode

    // Apply dark mode class to document
    useEffect(() => {
        if (darkMode) {
            document.documentElement.classList.add('dark');
        } else {
            document.documentElement.classList.remove('dark');
        }
    }, [darkMode]);

    useEffect(() => {
        checkServerStatus();

        // Listen for server logs
        let unlistenFn: (() => void) | null = null;

        listen<string>('server-log', (event) => {
            console.log('Received server-log event:', event.payload);
            const logLine = event.payload;
            addLog(logLine, 'server');

            // Initialize reasoning object if not exists
            if (!(window as any).__pendingReasoning) {
                (window as any).__pendingReasoning = {};
            }

            // Parse market regime FIRST (comes before signal in logs)
            if (
                logLine.includes('Market regime:') ||
                logLine.includes('Regime:')
            ) {
                const regimeMatch = logLine.match(/Regime:\s*(\w+)/i);
                if (regimeMatch) {
                    (window as any).__pendingReasoning.market_regime =
                        regimeMatch[1];
                }
            }

            // Parse filtered signal
            if (logLine.includes('Filtered from')) {
                const filteredMatch = logLine.match(/Filtered from\s*(\w+)/i);
                if (filteredMatch) {
                    (window as any).__pendingReasoning.filtered = true;
                    (window as any).__pendingReasoning.raw_signal =
                        filteredMatch[1].toLowerCase();
                }
            }

            // Parse signal from logs - format: "Signal: LONG"
            if (logLine.includes('Signal:')) {
                const signalMatch = logLine.match(
                    /Signal:\s*(LONG|SHORT|HOLD)/i
                );

                if (signalMatch) {
                    // Store signal temporarily, wait for confidence line
                    (window as any).__pendingSignal =
                        signalMatch[1].toLowerCase();
                }
            }

            // Parse confidence from logs - format: "Confidence: 0.8500 (85.00%)"
            if (
                logLine.includes('Confidence:') &&
                (window as any).__pendingSignal
            ) {
                const confidenceMatch = logLine.match(/Confidence:\s*([\d.]+)/);

                if (confidenceMatch) {
                    setLatestSignal({
                        signal: (window as any).__pendingSignal as
                            | 'long'
                            | 'short'
                            | 'hold',
                        confidence: parseFloat(confidenceMatch[1]),
                        timestamp: new Date(),
                        reasoning: (window as any).__pendingReasoning || {},
                    });
                    delete (window as any).__pendingSignal;
                    delete (window as any).__pendingReasoning;
                }
            }

            // Parse risk management parameters
            if (logLine.includes('Contracts:') && latestSignal) {
                const contractsMatch = logLine.match(/Contracts:\s*(\d+)/);
                if (contractsMatch) {
                    (window as any).__riskContracts = parseInt(
                        contractsMatch[1]
                    );
                }
            }
            if (logLine.includes('Entry Price:') && latestSignal) {
                const entryMatch = logLine.match(/Entry Price:\s*\$?([\d.]+)/);
                if (entryMatch) {
                    (window as any).__riskEntry = parseFloat(entryMatch[1]);
                }
            }
            if (logLine.includes('Stop Loss:') && latestSignal) {
                const stopMatch = logLine.match(/Stop Loss:\s*\$?([\d.]+)/);
                if (stopMatch) {
                    (window as any).__riskStop = parseFloat(stopMatch[1]);
                }
            }
            if (logLine.includes('Take Profit:') && latestSignal) {
                const tpMatch = logLine.match(/Take Profit:\s*\$?([\d.]+)/);
                if (tpMatch && (window as any).__riskContracts) {
                    // We have all risk parameters, update the signal
                    setLatestSignal((prev) =>
                        prev
                            ? {
                                  ...prev,
                                  risk_management: {
                                      contracts: (window as any)
                                          .__riskContracts,
                                      entry_price: (window as any).__riskEntry,
                                      stop_loss: (window as any).__riskStop,
                                      take_profit: parseFloat(tpMatch[1]),
                                  },
                              }
                            : null
                    );
                    // Clean up
                    delete (window as any).__riskContracts;
                    delete (window as any).__riskEntry;
                    delete (window as any).__riskStop;
                }
            }
        })
            .then((fn) => {
                unlistenFn = fn;
                console.log('Server log listener registered successfully');
            })
            .catch((err) => {
                console.error('Failed to register server log listener:', err);
            });

        return () => {
            if (unlistenFn) {
                unlistenFn();
            }
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    // Removed auto-scroll - user doesn't want it

    // Poll server data when server is running
    useEffect(() => {
        if (!serverRunning) return;

        // Initial fetch
        fetchServerData();

        // Poll every 2 seconds
        const interval = setInterval(fetchServerData, 2000);

        return () => clearInterval(interval);
    }, [serverRunning]);

    const addLog = (message: string, type: LogEntry['type'] = 'info') => {
        setLogs((prev) => [...prev, { message, type, timestamp: new Date() }]);
    };

    const checkServerStatus = async () => {
        try {
            const running = await invoke<boolean>('check_server_status');
            console.log('[DEBUG] Server running status:', running);
            setServerRunning(running);
        } catch (error) {
            console.error('Failed to check server status:', error);
        }
    };

    const startServer = async () => {
        setLoading(true);
        try {
            const result = await invoke<string>('start_server');
            addLog(result, 'success');
            setServerRunning(true);
        } catch (error) {
            addLog(`Failed to start server: ${error}`, 'error');
        } finally {
            setLoading(false);
        }
    };

    const stopServer = async () => {
        setLoading(true);
        try {
            const result = await invoke<string>('stop_server');
            addLog(result, 'info');
            setServerRunning(false);
            // Clear server data when stopping
            setHealthStatus(null);
            setTrainingStatus(null);
            setLatestSignal(null);
        } catch (error) {
            addLog(`Failed to stop server: ${error}`, 'error');
        } finally {
            setLoading(false);
        }
    };

    const testServer = async () => {
        setLoading(true);
        try {
            const result = await invoke<string>('test_server');
            addLog(result, 'success');
        } catch (error) {
            addLog(`Server test failed: ${error}`, 'error');
        } finally {
            setLoading(false);
        }
    };

    const fetchServerData = async () => {
        if (!serverRunning) {
            setTrainingStatus(null);
            setHealthStatus(null);
            return;
        }

        console.log('[DEBUG] Fetching server data...');

        try {
            // Fetch health status with cache-busting
            const healthResponse = await fetch(
                `http://127.0.0.1:8000/health-check?t=${Date.now()}`,
                { cache: 'no-store' }
            );
            console.log(
                '[DEBUG] Health response status:',
                healthResponse.status
            );
            if (healthResponse.ok) {
                const health = await healthResponse.json();
                console.log('[DEBUG] Health data:', health);
                setHealthStatus(health);
            }
        } catch (error) {
            console.error('Failed to fetch health status:', error);
        }

        try {
            // Fetch training status with cache-busting
            const trainingResponse = await fetch(
                `http://127.0.0.1:8000/training-status?t=${Date.now()}`,
                { cache: 'no-store' }
            );
            console.log(
                '[DEBUG] Training response status:',
                trainingResponse.status
            );
            if (trainingResponse.ok) {
                const training = await trainingResponse.json();
                console.log('[DEBUG] Training data:', training);
                setTrainingStatus(training);
            }
        } catch (error) {
            console.error('Failed to fetch training status:', error);
        }
    };

    const clearLogs = () => {
        setLogs([]);
    };

    return (
        <div className='min-h-screen bg-gradient-to-br from-purple-50 via-white to-blue-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900 p-6'>
            <div className='max-w-4xl mx-auto space-y-6'>
                {/* Header */}
                <div className='text-center space-y-2 relative'>
                    {/* Dark Mode Toggle */}
                    <div className='absolute right-0 top-0'>
                        <Button
                            onClick={() => setDarkMode(!darkMode)}
                            variant='ghost'
                            size='icon'
                            className='rounded-full'
                        >
                            {darkMode ? (
                                <Sun className='w-5 h-5' />
                            ) : (
                                <Moon className='w-5 h-5' />
                            )}
                        </Button>
                    </div>

                    <div className='flex items-center justify-center gap-2'>
                        <Server className='w-8 h-8 text-primary' />
                        <h1 className='text-4xl font-bold bg-gradient-to-r from-purple-600 to-blue-600 dark:from-purple-400 dark:to-blue-400 bg-clip-text text-transparent'>
                            RNN Trading Server
                        </h1>
                    </div>
                    <p className='text-muted-foreground'>
                        Manage your AI-powered trading server
                    </p>
                </div>

                {/* Control Button */}
                <Card>
                    <CardContent className='pt-6'>
                        <Button
                            onClick={serverRunning ? stopServer : startServer}
                            disabled={loading}
                            variant={serverRunning ? 'destructive' : 'default'}
                            className='w-full'
                            size='lg'
                        >
                            {loading ? (
                                <Loader2 className='w-4 h-4 mr-2 animate-spin' />
                            ) : serverRunning ? (
                                <Square className='w-4 h-4 mr-2' />
                            ) : (
                                <Play className='w-4 h-4 mr-2' />
                            )}
                            {serverRunning ? 'Stop Server' : 'Start Server'}
                        </Button>
                    </CardContent>
                </Card>

                {/* Dashboard Grid */}
                {serverRunning && (
                    <div className='grid grid-cols-1 md:grid-cols-2 gap-6'>
                        {/* Server Health Card */}
                        <Card>
                            <CardHeader>
                                <CardTitle className='flex items-center gap-2'>
                                    <Server className='w-5 h-5' />
                                    Server Health
                                </CardTitle>
                            </CardHeader>
                            <CardContent className='space-y-3'>
                                <div className='flex items-center justify-between'>
                                    <span className='text-sm text-muted-foreground'>
                                        Status
                                    </span>
                                    <Badge
                                        variant={
                                            healthStatus?.status === 'ok'
                                                ? 'default'
                                                : 'destructive'
                                        }
                                    >
                                        {healthStatus?.status === 'ok' ? (
                                            <CheckCircle2 className='w-3 h-3 mr-1' />
                                        ) : (
                                            <XCircle className='w-3 h-3 mr-1' />
                                        )}
                                        {healthStatus?.status || 'Unknown'}
                                    </Badge>
                                </div>
                                <div className='flex items-center justify-between'>
                                    <span className='text-sm text-muted-foreground'>
                                        Model Trained
                                    </span>
                                    {healthStatus?.model_trained ? (
                                        <CheckCircle2 className='w-5 h-5 text-green-500' />
                                    ) : (
                                        <XCircle className='w-5 h-5 text-red-500' />
                                    )}
                                </div>
                                <div className='flex items-center justify-between'>
                                    <span className='text-sm text-muted-foreground'>
                                        Device
                                    </span>
                                    <span className='text-sm font-mono'>
                                        {healthStatus?.device || 'N/A'}
                                    </span>
                                </div>
                            </CardContent>
                        </Card>

                        {/* Training Status Card */}
                        <Card>
                            <CardHeader>
                                <CardTitle className='flex items-center gap-2'>
                                    <Brain className='w-5 h-5' />
                                    Training Status
                                </CardTitle>
                            </CardHeader>
                            <CardContent className='space-y-3'>
                                <div className='flex items-center justify-between'>
                                    <span className='text-sm text-muted-foreground'>
                                        Status
                                    </span>
                                    <Badge
                                        variant={
                                            trainingStatus?.is_training
                                                ? 'default'
                                                : trainingStatus?.progress?.includes(
                                                      'complete'
                                                  ) ||
                                                  trainingStatus?.progress?.includes(
                                                      'Complete'
                                                  )
                                                ? 'default'
                                                : 'secondary'
                                        }
                                        className={
                                            trainingStatus?.progress?.includes(
                                                'complete'
                                            ) ||
                                            trainingStatus?.progress?.includes(
                                                'Complete'
                                            )
                                                ? 'bg-green-500 hover:bg-green-600'
                                                : ''
                                        }
                                    >
                                        {trainingStatus?.is_training ? (
                                            <>
                                                <Loader2 className='w-3 h-3 mr-1 animate-spin' />
                                                Training
                                            </>
                                        ) : trainingStatus?.progress?.includes(
                                              'complete'
                                          ) ||
                                          trainingStatus?.progress?.includes(
                                              'Complete'
                                          ) ? (
                                            <>
                                                <CheckCircle2 className='w-3 h-3 mr-1' />
                                                Complete
                                            </>
                                        ) : (
                                            'Idle'
                                        )}
                                    </Badge>
                                </div>
                                {trainingStatus?.progress && (
                                    <div className='space-y-1'>
                                        <span className='text-sm text-muted-foreground'>
                                            Progress
                                        </span>
                                        <p className='text-sm font-medium'>
                                            {trainingStatus.progress}
                                        </p>
                                    </div>
                                )}
                                {trainingStatus?.error && (
                                    <div className='p-2 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded'>
                                        <p className='text-xs text-red-800 dark:text-red-200'>
                                            {trainingStatus.error}
                                        </p>
                                    </div>
                                )}
                            </CardContent>
                        </Card>

                        {/* Latest Signal Card */}
                        <Card className='md:col-span-2'>
                            <CardHeader>
                                <CardTitle className='flex items-center gap-2'>
                                    <Activity className='w-5 h-5' />
                                    Latest Signal
                                </CardTitle>
                            </CardHeader>
                            <CardContent>
                                {latestSignal ? (
                                    <div className='space-y-4'>
                                        <div className='flex items-center justify-between'>
                                            <div className='flex items-center gap-3'>
                                                {latestSignal.signal ===
                                                'long' ? (
                                                    <TrendingUp className='w-8 h-8 text-green-500' />
                                                ) : latestSignal.signal ===
                                                  'short' ? (
                                                    <TrendingDown className='w-8 h-8 text-red-500' />
                                                ) : (
                                                    <Minus className='w-8 h-8 text-gray-500' />
                                                )}
                                                <div>
                                                    <p className='text-2xl font-bold uppercase'>
                                                        {latestSignal.signal}
                                                    </p>
                                                    <p className='text-sm text-muted-foreground'>
                                                        {latestSignal.timestamp.toLocaleTimeString()}
                                                    </p>
                                                </div>
                                            </div>
                                            <div className='text-right'>
                                                <p className='text-sm text-muted-foreground'>
                                                    Confidence
                                                </p>
                                                <p className='text-2xl font-bold'>
                                                    {(
                                                        latestSignal.confidence *
                                                        100
                                                    ).toFixed(1)}
                                                    %
                                                </p>
                                            </div>
                                        </div>

                                        {/* Reasoning Section */}
                                        {latestSignal.reasoning && (
                                            <div className='p-3 bg-muted/50 rounded-lg space-y-2'>
                                                <p className='text-xs font-semibold text-muted-foreground uppercase'>
                                                    Signal Reasoning
                                                </p>
                                                <div className='space-y-1 text-sm'>
                                                    {latestSignal.reasoning
                                                        .market_regime && (
                                                        <p>
                                                            <span className='font-medium'>
                                                                Market Regime:
                                                            </span>{' '}
                                                            <span className='capitalize'>
                                                                {
                                                                    latestSignal
                                                                        .reasoning
                                                                        .market_regime
                                                                }
                                                            </span>
                                                        </p>
                                                    )}
                                                    {latestSignal.reasoning
                                                        .filtered &&
                                                        latestSignal.reasoning
                                                            .raw_signal && (
                                                            <p className='text-amber-600 dark:text-amber-400'>
                                                                <span className='font-medium'>
                                                                    Filtered:
                                                                </span>{' '}
                                                                Original signal
                                                                was{' '}
                                                                {latestSignal.reasoning.raw_signal.toUpperCase()}
                                                                , filtered to{' '}
                                                                {latestSignal.signal.toUpperCase()}{' '}
                                                                due to low
                                                                confidence
                                                            </p>
                                                        )}
                                                    {latestSignal.reasoning
                                                        .mtf_filtered && (
                                                        <p className='text-amber-600 dark:text-amber-400'>
                                                            <span className='font-medium'>
                                                                MTF Filter:
                                                            </span>{' '}
                                                            Multi-timeframe
                                                            alignment check
                                                            blocked
                                                            counter-trend trade
                                                        </p>
                                                    )}
                                                    {latestSignal.reasoning
                                                        .regime_filtered && (
                                                        <p className='text-amber-600 dark:text-amber-400'>
                                                            <span className='font-medium'>
                                                                Regime Filter:
                                                            </span>{' '}
                                                            Market conditions
                                                            not favorable for
                                                            trading
                                                        </p>
                                                    )}
                                                    {latestSignal.reasoning
                                                        .confidence_threshold && (
                                                        <p>
                                                            <span className='font-medium'>
                                                                Threshold:
                                                            </span>{' '}
                                                            {(
                                                                latestSignal
                                                                    .reasoning
                                                                    .confidence_threshold *
                                                                100
                                                            ).toFixed(1)}
                                                            %
                                                        </p>
                                                    )}
                                                </div>
                                            </div>
                                        )}

                                        {latestSignal.risk_management && (
                                            <div className='grid grid-cols-2 md:grid-cols-4 gap-3 pt-3 border-t'>
                                                <div>
                                                    <p className='text-xs text-muted-foreground'>
                                                        Contracts
                                                    </p>
                                                    <p className='text-sm font-semibold'>
                                                        {
                                                            latestSignal
                                                                .risk_management
                                                                .contracts
                                                        }
                                                    </p>
                                                </div>
                                                <div>
                                                    <p className='text-xs text-muted-foreground'>
                                                        Entry
                                                    </p>
                                                    <p className='text-sm font-semibold'>
                                                        $
                                                        {latestSignal.risk_management.entry_price.toFixed(
                                                            2
                                                        )}
                                                    </p>
                                                </div>
                                                <div>
                                                    <p className='text-xs text-muted-foreground'>
                                                        Stop Loss
                                                    </p>
                                                    <p className='text-sm font-semibold text-red-500'>
                                                        $
                                                        {latestSignal.risk_management.stop_loss.toFixed(
                                                            2
                                                        )}
                                                    </p>
                                                </div>
                                                <div>
                                                    <p className='text-xs text-muted-foreground'>
                                                        Take Profit
                                                    </p>
                                                    <p className='text-sm font-semibold text-green-500'>
                                                        $
                                                        {latestSignal.risk_management.take_profit.toFixed(
                                                            2
                                                        )}
                                                    </p>
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                ) : (
                                    <div className='text-center py-8 text-muted-foreground'>
                                        <AlertCircle className='w-12 h-12 mx-auto mb-2 opacity-50' />
                                        <p>No signals yet</p>
                                        <p className='text-sm'>
                                            Waiting for trading data...
                                        </p>
                                    </div>
                                )}
                            </CardContent>
                        </Card>
                    </div>
                )}

                {/* Logs Card - Compact */}
                {serverRunning && logs.length > 0 && (
                    <Card>
                        <CardHeader>
                            <div className='flex items-center justify-between'>
                                <CardTitle className='text-base'>
                                    Server Logs
                                </CardTitle>
                                <Button
                                    onClick={clearLogs}
                                    variant='ghost'
                                    size='sm'
                                >
                                    Clear
                                </Button>
                            </div>
                        </CardHeader>
                        <CardContent>
                            <div className='bg-gray-900 rounded-lg p-3 h-48 overflow-y-auto font-mono text-xs'>
                                <div className='space-y-0.5'>
                                    {logs.slice(-50).map((log, index) => (
                                        <div
                                            key={`${log.timestamp.getTime()}-${index}`}
                                            className={`${
                                                log.type === 'error'
                                                    ? 'text-red-400'
                                                    : log.type === 'success'
                                                    ? 'text-green-400'
                                                    : log.type === 'server'
                                                    ? 'text-blue-300'
                                                    : 'text-gray-300'
                                            }`}
                                        >
                                            {log.message}
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </CardContent>
                    </Card>
                )}

                {/* Footer */}
                <div className='text-center text-sm text-muted-foreground'>
                    <p>
                        Built with Tauri, React, and TypeScript •{' '}
                        <a
                            href='https://github.com/rrhodeswebdev/ninjatrader-8'
                            className='underline hover:text-foreground'
                            target='_blank'
                            rel='noopener noreferrer'
                        >
                            View on GitHub
                        </a>
                    </p>
                </div>
            </div>
        </div>
    );
}

export default App;
